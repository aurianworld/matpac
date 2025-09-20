# parts from https://github.com/facebookresearch/dino and https://github.com/nttcslab/m2d


import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from typing import Optional


from fairseq.modules import EMAModule, EMAModuleConfig

from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model

from omegaconf import II

from ..tasks.pretraining import PretrainingConfig, LogmelPretrainingTask

from .encoder import (
    encoder_layers_config,
    encoder_layers
)

from .decoder import (
    decoder_config,
    decoder
)

from .mcl import (
    mcl_config,
    mcl_heads
)

from .utils import (
    PatchEmbed,
    get_annealed_rate,
    random_unstructured_mask,
    get_2d_sincos_pos_embed,
    get_exp_annealed_rate
)

from .cls_head import (Cls_Head_Config, Cls_Head, Cls_Loss)

logger = logging.getLogger(__name__)


@dataclass
class general_config(FairseqDataclass):

  ## Model Parameters ##
  encoder: encoder_layers_config = encoder_layers_config()

  decoder: decoder_config = decoder_config()

  use_mcl: bool = False
  mcl_heads: mcl_config = mcl_config()

  ####  Pred param ###
  ## Masking ##
  mask_ratio: float = 0.7
  return_layers: bool = False

  ## Logmel Spec Shape ##
  n_freq: int = 80
  n_t: int = II("task.max_sample_size")
  patch_size: int = II("task.patch_size")

  ## EMA Parameters ##
  ema_decay: float = field(default=0.99995, metadata={
                           "help": "initial ema decay rate"})
  log_norms: bool = True
  ema_end_decay: float = field(
      default=0.99999, metadata={"help": "final ema decay rate"}
  )
  ema_anneal_end_step: int = II("optimization.max_update")

  ### Cls Head param  ###
  cls_head_cfg: Optional[Cls_Head_Config] = Cls_Head_Config()

  ### Loss param ###
  alpha: float = 0.5


@register_model("matpac", dataclass=general_config)
class GeneralModel(BaseFairseqModel):
  def __init__(self,
               cfg: general_config,
               task_cfg: PretrainingConfig,
               ):
    super().__init__()
    logger.info(f"General model Config: {cfg}")

    self.cfg = cfg
    self.task_cfg = task_cfg

    ## General parameters ##
    self.patch_embed = PatchEmbed(img_size=[cfg.n_freq, cfg.n_t],
                                  patch_size=cfg.patch_size,
                                  embed_dim=cfg.encoder.embed_dim,
                                  )
    num_patches = self.patch_embed.num_patches

    self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.encoder.embed_dim))
    self.pos_embed = nn.Parameter(
        torch.zeros(
            1, num_patches + 1, cfg.encoder.embed_dim),
        requires_grad=False
    )
    self.return_layers = self.cfg.return_layers

    ## Student/Main Encoder  ##
    self.student_encoder = encoder_layers(cfg=cfg.encoder)

    ## MCL Heads ##
    if self.cfg.use_mcl:
      self.mcl_heads = mcl_heads(cfg=cfg.mcl_heads)

    ## Decoder ##
    self.decoder = decoder(cfg=cfg.decoder,
                           num_patches=num_patches)

    ## Teacher Encoder ##
    self.teacher_encoder = self.make_ema_teacher(
        cfg.ema_decay)

    # Cls head Init
    self.set_cls_head_init(cfg.cls_head_cfg,
                           )

    ## Trackers and init weights and other functions/utils ##
    self.num_updates = 0
    self._random_mask_fn = random_unstructured_mask
    self.initialize_weights()

  #################################
  #################################
  ######## GENERAL FORWARD ########
  #################################
  #################################

  def forward(self,
              **net_input):
    result = {
        "losses": {},
        "ema_decay": None
    }

    # Putting teacher model on same device as the other blocks
    self.teacher_to_device(net_input["source"], self.teacher_encoder)

    if not self.training:
      result = self.forward_validation(**net_input, result=result)
      return result

    loss_cls, loss_pred, result_cls_pred = self.forward_cls_pred(**net_input)

    result = {**result, **result_cls_pred}

    # We avoid to compute gradients if we do not use cls or pred loss.
    if self.cfg.alpha == 0:
      result["losses"]["loss_cls"] = loss_cls
    elif self.cfg.alpha == 1:
      result["losses"]["loss_pred"] = loss_pred.view(-1)
    else:
      result["losses"]["loss_cls"] = loss_cls
      result["losses"]["loss_pred"] = loss_pred.view(-1)

    result["sample_size"] = loss_cls.numel()

    # Logging teacher stuffs
    for k, v in self.teacher_encoder.logs.items():
      result[k] = v
    result["ema_decay"] = self.teacher_encoder.get_decay() * 1000
    return result

  #################################
  #################################
  ##### FORWARD validation  #######
  #################################
  #################################

  def forward_validation(self, source, result):

    x = self.patch_and_pos_emb(source=source)

    # We use only a slice of all frequency to have fast computation
    x = x[:, 0:(self.cfg.n_freq//16 * 2), :]

    # adding cls token for forward
    cls_token = self.cls_token + self.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    # Forward and dropping classification token for pred
    z_t = self.teacher_encoder.model.forward(x)
    z_t = self.drop_cls_token(z_t)

    z_s = self.student_encoder.forward(x)
    z_s = self.drop_cls_token(z_s)

    # Forward for cls
    p_s = self.cls_student_head(z_s)
    p_t = self.cls_teacher_head.model(z_t)

    # centering teacher
    teacher_temp = get_annealed_rate(
        start=self.cls_head_cfg.warmup_teacher_temp,
        end=self.cls_head_cfg.teacher_temp,
        curr_step=self.num_updates,
        total_steps=self.cls_head_cfg.warmup_n_steps,
    )
    p_t = F.softmax((p_t - self.cls_head_loss.center) / teacher_temp, dim=-1)

    p_s = p_s / self.cls_head_loss.student_temp
    p_s = p_s.view((-1, p_s.shape[-1]))
    p_t = p_t.view(
        (-1, p_t.shape[-1]))

    loss_cls = -torch.sum(
        p_t * F.log_softmax(p_s, dim=-1),
        dim=-1
    )

    # loss on the prediction and target in latent space
    target = torch.nn.functional.normalize(z_t, dim=-1, p=2)
    pred = torch.nn.functional.normalize(z_s, dim=-1, p=2)

    result["var_z_hat_m"] = self.compute_var(pred) * 1000
    result["var_z_m"] = self.compute_var(target) * 1000

    loss_pred = target * pred
    loss_pred = 2 - 2 * loss_pred.sum(dim=-1)

    if self.cfg.alpha == 0:
      result["losses"]["loss_cls"] = loss_cls
    elif self.cfg.alpha == 1:
      result["losses"]["loss_pred"] = loss_pred.view(-1)
    else:
      result["losses"]["loss_cls"] = loss_cls
      result["losses"]["loss_pred"] = loss_pred.view(-1)
    result["sample_size"] = loss_cls.numel()

    return result

  #################################
  #################################
  ##### FORWARD CLS AND PRED ######
  #################################
  #################################

  def forward_cls_pred(self, source):

    result = {}

    z_v, x_m, mask, ids_restore = self.forward_student_encoder_pred(
        source)
    z_hat_m = self.forward_decoder(z_v, ids_restore)  # [N, targL, D]
    with torch.no_grad():
      z_m = self.forward_teacher_encoder_pred(x_m,
                                              drop_cls=False)

      # best_head_ix is None when we do not train with MCL heads
      loss_pred, best_head_idx, result = self.forward_loss_pred(self.drop_cls_token(z_m),
                                                                z_hat_m,
                                                                result)

    # We get the temperature tau for the teacher cls head
    teacher_temp = get_annealed_rate(
        start=self.cls_head_cfg.warmup_teacher_temp,
        end=self.cls_head_cfg.teacher_temp,
        curr_step=self.num_updates,
        total_steps=self.cls_head_cfg.warmup_n_steps,
    )
    result["t_temp"] = teacher_temp*100

    # If we are using MCL we chose the best prediction to be used in the CLS Head
    if self.cfg.use_mcl:
      _, _, n, D = z_hat_m.shape
      z_hat_m = torch.gather(
          z_hat_m,
          dim=2,
          index=best_head_idx.unsqueeze(-1).unsqueeze(dim=-1).repeat(1,
                                                                     1,
                                                                     1,
                                                                     D))

      z_hat_m = z_hat_m.squeeze(dim=2)

    # First we put the teacher on the right device and type
    self.teacher_to_device(z_hat_m, self.cls_teacher_head)

    # Selecting the patchs token of the sequence.
    z_m = self.drop_cls_token(z_m)

    # The classification are projected with the Cls head.
    p_hat_m = self.cls_student_head(z_hat_m)
    p_m = self.cls_teacher_head.model(z_m)

    # Sharpening and Centering of the teacher output
    p_m = self.cls_head_loss.softmax_center_teacher(p_m,
                                                    teacher_temp)

    # Updating the center
    self.cls_head_loss.update_center(p_m
                                     )

    # Computing the Classification Loss
    loss_cls, result = self.cls_head_loss(p_hat_m,
                                          p_m,
                                          result)

    self.do_ema_step(teacher_model=self.cls_teacher_head,
                     student_model=self.cls_student_head,
                     ema_decay=self.cls_head_cfg.ema_decay,
                     ema_end_decay=self.cls_head_cfg.ema_end_decay,
                     ema_anneal_end_step=self.cfg.ema_anneal_end_step,)

    self.do_ema_step(teacher_model=self.teacher_encoder,
                     student_model=self.student_encoder,
                     ema_decay=self.cfg.ema_decay,
                     ema_end_decay=self.cfg.ema_end_decay,
                     ema_anneal_end_step=self.cfg.ema_anneal_end_step)

    return loss_cls, loss_pred, result

  def set_cls_head_init(self,
                        cls_head_cfg):
    """Function that adds all the necessary variables and modules to work with
    Classification loss"""
    self.cls_head_cfg = cls_head_cfg

    self.cls_head_loss = Cls_Loss(out_dim=self.cls_head_cfg.head_dim_out)

    self.cls_student_head, self.cls_teacher_head = self.make_cls_heads()

  @torch.no_grad()
  def make_cls_heads(self):
    """Create the student and teacher classification heads and set the ema
    between them.
    """

    student_head = Cls_Head(in_dim=self.cfg.encoder.embed_dim,
                            hidden_dim=self.cls_head_cfg.hidden_dim,
                            out_dim=self.cls_head_cfg.head_dim_out)
    teacher_head_copy = Cls_Head(in_dim=self.cfg.encoder.embed_dim,
                                 hidden_dim=self.cls_head_cfg.hidden_dim,
                                 out_dim=self.cls_head_cfg.head_dim_out)

    ema_config = EMAModuleConfig(
        ema_decay=self.cls_head_cfg.ema_decay,
        ema_fp32=True,
        log_norms=self.cfg.log_norms,
        add_missing_params=False,
    )

    for p_s, p_t in zip(student_head.parameters(),
                        teacher_head_copy.parameters()):
      p_t.data.copy_(p_s.data)

    teacher_head_copy.requires_grad_(False)

    teacher_head_copy_ema = EMAModule(
        teacher_head_copy,
        ema_config,
        copy_model=False,
    )

    return student_head, teacher_head_copy_ema

  #############################
  #############################
  ####### FORWARD PRED ########
  #############################
  #############################

  def forward_student_encoder_pred(self, source):
    """Does the forward pass in the student encoder and does the masking
    """

    # Patch Logmel Spectrogram
    x = self.patch_embed(source.unsqueeze(dim=1))

    # Add positional
    pos_embed = self.pos_embed[:, 1:, :]
    if x.shape[1] < pos_embed.shape[1]:
      # audio: shorten pos_embed for a short input
      dims = pos_embed.shape[-1]
      fbins = self.grid_size()[0]
      frames = x.shape[1] // fbins
      pos_embed = pos_embed.reshape(
          1, fbins, -1, dims)[:, :, :frames, :].reshape(1, fbins*frames, dims)
    x = x + pos_embed

    # Masking
    x, x_targ, mask, ids_restore = self.random_masking(x, self.cfg.mask_ratio)

    # append cls token
    cls_token = self.cls_token + self.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    x = self.student_encoder.forward(x, return_layers=False)

    return x, x_targ, mask, ids_restore

  def forward_teacher_encoder_pred(self,
                                   x_targ,
                                   drop_cls=True):
    # append cls token
    cls_token = self.cls_token + self.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(x_targ.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x_targ), dim=1)

    # apply Transformer blocks
    for l, blk in enumerate(self.teacher_encoder.model.blocks):
      x = blk(x)

    x = self.teacher_encoder.model.norm(x)

    # remove cls token
    if drop_cls:
      x = self.drop_cls_token(x)

    return x

  def forward_loss_pred(self, target, pred, result):
    """
    target: [N, targL, D]
    pred: [N, targL, D]
    """

    target = torch.nn.functional.normalize(target, dim=-1, p=2)
    pred = torch.nn.functional.normalize(pred, dim=-1, p=2)

    if self.cfg.use_mcl:
      loss = target.unsqueeze(dim=2) * pred
      for i in range(self.cfg.mcl_heads.n_head):
        n = f"var_z_hat_m_{i}"
        result[n] = self.compute_var(pred[:, :, i, :].float()) * 1000
      result["var_z_m"] = self.compute_var(target) * 1000

    else:
      loss = target * pred
      result["var_z_hat_m"] = self.compute_var(pred) * 1000
      result["var_z_m"] = self.compute_var(target) * 1000

    loss = 2 - 2 * loss.sum(dim=-1)

    if not self.cfg.use_mcl:
      return loss, None, result

    else:
      best_head_idx = loss.min(dim=-1)[1]

      temperature = get_exp_annealed_rate(start=self.cfg.mcl_heads.temperature_start,
                                          decay_rate=self.cfg.mcl_heads.decay_rate,
                                          curr_step=self.num_updates)

      # Exponential annealed mcl
      if temperature > self.cfg.mcl_heads.temperature_end:
        # We put a negative sign because we want the smallest loss not the biggest
        soft_assignation = (torch.softmax(-loss/temperature, dim=2)).detach()
        loss_mcl_pred = soft_assignation * loss
        loss_mcl_pred = loss_mcl_pred.sum(dim=-1)

        result["mcl_temp"] = temperature * 1000

      # Switching to regular mcl if temperature to low (to avoid small values)
      else:
        loss_mcl_pred = loss.min(dim=-1)[0]

        result["mcl_temp"] = self.cfg.mcl_heads.temperature_end * 1000

      return loss_mcl_pred, best_head_idx, result

  def forward_decoder(self, x,
                      ids_restore,
                      keep_cls=False):
    len_keep = x.shape[1] - 1  # tokens - cls

    # embed tokens
    x = self.decoder.decoder_embed(x)
    D = x.shape[-1]

    # append mask tokens to sequence
    mask_tokens = self.decoder.mask_token.repeat(
        x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
    x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
    x_ = torch.gather(
        x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  # unshuffle
    x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

    # add pos embed
    x = x + self.decoder.decoder_pos_embed

    # apply Transformer blocks
    for blk in self.decoder.decoder_blocks:
      x = blk(x)
    x = self.decoder.decoder_norm(x)

    # remove cls token
    y = self.drop_cls_token(x)
    # re-shuffle, and keep prediction only
    ids_shuffle = torch.argsort(ids_restore, dim=1)
    y = torch.gather(
        y, dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, y.shape[-1]))
    y = y[:, len_keep:]  # prediction only

    # append cls if needed
    if keep_cls:
      y = torch.cat([x[:, :1, :], y], dim=1)

    if self.cfg.use_mcl:
      y, _ = self.mcl_heads(y)

    else:
      y = self.decoder.decoder_pred(y)

    return y

  ######################
  ######################
  ####### UTILS ########
  ######################
  ######################

  @staticmethod
  def compute_var(y):
    y = y.view(-1, y.size(-1)).float()
    if dist.is_initialized():
      zc = torch.tensor(y.size(0)).cuda()
      zs = y.sum(dim=0)
      zss = (y**2).sum(dim=0)

      dist.all_reduce(zc)
      dist.all_reduce(zs)
      dist.all_reduce(zss)

      var = zss / (zc - 1) - (zs**2) / (zc * (zc - 1))
      return torch.sqrt(var + 1e-6).mean()
    else:
      return torch.sqrt(y.var(dim=0) + 1e-6).mean()

  def patch_and_pos_emb(self, source):
    # Patch Logmel Spectrogram
    x = self.patch_embed(source.unsqueeze(dim=1))

    # Add positional
    pos_embed = self.pos_embed[:, 1:, :]
    if x.shape[1] < pos_embed.shape[1]:
      # audio: shorten pos_embed for a short input
      dims = pos_embed.shape[-1]
      fbins = self.grid_size()[0]
      frames = x.shape[1] // fbins
      pos_embed = pos_embed.reshape(
          1, fbins, -1, dims)[:, :, :frames, :].reshape(1, fbins*frames, dims)
    x = x + pos_embed
    return x

  def extract_features(self, x, use_teacher=False):
    # Patch Logmel Spectrogram
    if x.ndim <= 3:
      x = x.unsqueeze(dim=1)
    x = self.patch_embed(x)

    # Add positional
    pos_embed = self.pos_embed[:, 1:, :]
    if x.shape[1] < pos_embed.shape[1]:
      # audio: shorten pos_embed for a short input
      dims = pos_embed.shape[-1]
      fbins = self.grid_size()[0]
      frames = x.shape[1] // fbins
      pos_embed = pos_embed.reshape(
          1, fbins, -1, dims)[:, :, :frames, :].reshape(1, fbins*frames, dims)
    x = x + pos_embed

    # append cls token
    cls_token = self.cls_token + self.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    if use_teacher:
      self.teacher_to_device(x, self.teacher_encoder)
      x = self.teacher_encoder.model.forward(x, return_layers=True)
    else:
      x = self.student_encoder.forward(x, return_layers=True)

    return x[:, -1, :, :], x  # Return embedding and layer results

  def all_layer_extract(self, x, cls_head=False, use_teacher=False):

    # Forward encoder
    out_enc, layer_results_enc = self.extract_features(
        x, use_teacher=use_teacher)

    # Forward decoder
    in_dec = self.decoder.decoder_embed(out_enc)

    # add pos embed
    pos_embed = self.decoder.decoder_pos_embed[:, 1:, :]
    if in_dec.shape[1] - 1 < pos_embed.shape[1]:
      # audio: shorten pos_embed for a short input
      dims = pos_embed.shape[-1]
      fbins = self.grid_size()[0]
      frames = in_dec.shape[1] // fbins
      pos_embed = pos_embed.reshape(
          1, fbins, -1, dims)[:, :, :frames, :].reshape(1, fbins*frames, dims)
    pos_embed = torch.concatenate((self.decoder.decoder_pos_embed[:, :1, :],
                                   pos_embed),
                                  dim=1)
    in_dec = in_dec + pos_embed

    # apply decoder Transformer blocks
    layers = []
    for blk in self.decoder.decoder_blocks:
      in_dec = blk(in_dec)
      layers.append(in_dec.unsqueeze(dim=1))
    out_dec = self.decoder.decoder_norm(in_dec)
    layers[-1] = out_dec.unsqueeze(dim=1)
    layer_results_dec = torch.cat(layers, dim=1)

    all_layers_out = {"layer_results_enc": layer_results_enc,
                      "layer_results_dec": layer_results_dec}

    # Lastly apply the cls head
    if cls_head:
      out_cls_head = self.cls_student_head(out_dec)
      all_layers_out["out_cls_head"] = out_cls_head

    return all_layers_out

  def random_masking(self, x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    # Random mask
    HorF, WorT = self.grid_size()
    if L < HorF * WorT:
      # audio: shorten pos_embed for a short input
      WorT = L // HorF
    ids_shuffle, len_keep = self._random_mask_fn(
        (N, HorF, WorT), mask_ratio, x.device)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the visible patch (student encoder) indexes
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(
        x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # keep the rest (teacher encoder)
    ids_keep = ids_shuffle[:, len_keep:]
    x_masked2 = torch.gather(
        x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is masked, 1 is masked2
    mask = torch.ones([N, L], device=x.device, dtype=x.dtype)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, x_masked2, mask, ids_restore

  def grid_size(self):
    # This fails with timm 0.4.5 -> return self.patch_embed.grid_size
    # Workaround for avoid compatibility issue
    img_size = np.array(self.patch_embed.img_size)
    patch_size = np.array(self.patch_embed.patch_size)
    grid_size = img_size // patch_size
    return grid_size

  @torch.no_grad()
  def make_ema_teacher(self, ema_decay):
    ema_config = EMAModuleConfig(
        ema_decay=ema_decay,
        ema_fp32=True,
        log_norms=self.cfg.log_norms,
        add_missing_params=False,
    )

    teacher_model = encoder_layers(cfg=self.cfg.encoder)
    teacher_model.requires_grad_(False)
    teacher_model.blocks.apply(self._init_weights)
    teacher_model.norm.apply(self._init_weights)

    teacher_model = EMAModule(
        teacher_model,
        ema_config,
        copy_model=False,
    )

    return teacher_model

  # TO ADD EMA/TEACHERS TO CKPT
  def state_dict(self, destination=None, prefix="", keep_vars=False):
    state = super().state_dict(destination, prefix, keep_vars)

    if self.teacher_encoder is not None:
      state[prefix + "_ema"] = self.teacher_encoder.fp32_params

    if self.cls_teacher_head is not None:
      state[prefix + "_cls_head_ema"] = self.cls_teacher_head.fp32_params
    return state

  # TO RELOAD EMA/TEACHER TO CKPT
  def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
    k = prefix + "_ema"
    if self.teacher_encoder is not None:
      assert k in state_dict
      self.teacher_encoder.restore(state_dict[k], True)
      del state_dict[k]
    elif k in state_dict:
      del state_dict[k]

    k = prefix + "_cls_head_ema"
    if self.cls_teacher_head is not None:
      assert k in state_dict
      self.cls_teacher_head.restore(state_dict[k], True)
      del state_dict[k]
    elif k in state_dict:
      del state_dict[k]

    return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

  def teacher_to_device(self, x, model):
    # Teacher Processing
    p = next(model.model.parameters())
    device = x.device
    dtype = x.dtype
    ema_device = p.device
    ema_dtype = p.dtype

    if ema_device != device or ema_dtype != dtype:
      logger.info(
          f"adjusting teacher encoder dtype to {dtype} and device to {device}")
      model.model = model.model.to(
          dtype=dtype, device=device)
      ema_dtype = dtype

      def to_device(d):
        for k, p in d.items():
          if isinstance(d[k], dict):
            to_device(d[k])
          else:
            d[k] = p.to(device=device)

      to_device(model.fp32_params)

  def set_num_updates(self, num_updates):
    """Update the number of update"""
    super().set_num_updates(num_updates)

    self.num_updates = num_updates

  def do_ema_step(self, teacher_model,
                  student_model,
                  ema_decay,
                  ema_end_decay,
                  ema_anneal_end_step):
    num_updates = self.num_updates
    if teacher_model is not None and (
        self.num_updates == 0
    ):
      pass
    elif self.training and teacher_model is not None:
      ema_weight_decay = None
      if ema_decay != ema_end_decay:
        if num_updates >= ema_anneal_end_step:
          decay = self.cfg.ema_end_decay
        else:
          decay = get_annealed_rate(
              ema_decay,  # ema_decay,
              ema_end_decay,  # self.cfg.ema_end_decay,
              num_updates,
              ema_anneal_end_step  # self.cfg.ema_anneal_end_step,
          )
        teacher_model.set_decay(decay, weight_decay=ema_weight_decay)
      if teacher_model.get_decay() < 1:
        teacher_model.step(student_model)

  def initialize_weights(self):
    # initialization
    # initialize (and freeze) pos_embed by sin-cos embedding
    pos_embed = get_2d_sincos_pos_embed(
        self.pos_embed.shape[-1], self.grid_size(), cls_token=True)
    self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    decoder_pos_embed = get_2d_sincos_pos_embed(
        self.decoder.decoder_pos_embed.shape[-1], self.grid_size(), cls_token=True)
    self.decoder.decoder_pos_embed.data.copy_(
        torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

    # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
    w = self.patch_embed.proj.weight.data
    torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
    torch.nn.init.normal_(self.cls_token, std=.02)
    torch.nn.init.normal_(self.decoder.mask_token, std=.02)

    # initialize nn.Linear and nn.LayerNorm
    self.apply(self._init_weights)

  def _init_weights(self, m):
    if isinstance(m, nn.Linear):
      # we use xavier_uniform following official JAX ViT:
      torch.nn.init.xavier_uniform_(m.weight)
      if isinstance(m, nn.Linear) and m.bias is not None:
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
      nn.init.constant_(m.bias, 0)
      nn.init.constant_(m.weight, 1.0)

  @classmethod
  def build_model(cls,
                  cfg: general_config,
                  task: LogmelPretrainingTask):
    """Build a new model instance in Fairseq"""

    model = GeneralModel(cfg=cfg, task_cfg=task.cfg)
    return model

  def drop_cls_token(self, latent):
    # remove cls token [B, 1+H*W: D] -> [B, H*W, D]
    return latent[:, 1:, :]

  def get_cls_token(self, latent):
    # return cls token only [B, 1+H*W: D] -> [B, 1, D]
    return latent[:, :1, :]
