import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from einops import rearrange
import pandas as pd
import os
import json
from timm.models.layers import trunc_normal_

from matpac.preprocess import logMelSpectrogram
from matpac.encoder import encoder_layers_config, encoder_layers
from matpac.utils import PatchEmbed


@dataclass
class general_config:

  ## Model Parameters ##
  encoder: encoder_layers_config = encoder_layers_config()

  ## Logmel Spec Shape ##
  n_freq: int = 80
  n_t: int = 608
  patch_size: int = 16

  ## Audio Parameters ##
  sr: int = 16000

  ## Normalization params ##
  lms_mean: float = -7.056
  lms_std: float = 4.193


class matpac_wrapper(nn.Module):
  def __init__(self,
               inference_type="precise",  # or fast
               pull_time_dimension=True,
               as_class_head=False,
               concat_freq=True,
               cfg: general_config = general_config(),
               probe_checkpoint_paths: str = None
               ):
    super(matpac_wrapper, self).__init__()

    # Setting the parameters
    self.cfg = cfg
    self.inference_type = inference_type
    self.pull_time_dimension = pull_time_dimension
    self.as_class_head = as_class_head
    self.concat_freq = concat_freq

    # Setting the nn modules
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

    self.student_encoder = encoder_layers(cfg=cfg.encoder)

    # Setting the forward tools for pre-processing audios
    self.log_mel = logMelSpectrogram(sample_rate=cfg.sr,
                                     n_fft=400,
                                     win_length=0.025,
                                     hop_length=0.01,
                                     f_min=50,
                                     f_max=cfg.sr//2,
                                     log_offset=torch.finfo().eps,
                                     n_mels=cfg.n_freq,
                                     center=False)

    base_path = os.path.dirname(__file__)
    if as_class_head:

      self.head_norm = torch.nn.BatchNorm1d(
          cfg.encoder.embed_dim, affine=False)
      self.head = torch.nn.Linear(cfg.encoder.embed_dim, 527)
      trunc_normal_(self.head.weight, std=2e-5)

      csv_path = os.path.join(
          base_path, "class_mappings", "audioset_mapping.csv")
      # self.as_class_map = pd.read_csv(csv_path, index_col=None)
      self.class_map = pd.read_csv(
          csv_path).sort_values('mid').reset_index()
      self.class_map = self.class_map["display_name"].to_dict()

    if probe_checkpoint_paths is not None:

      if "openmic" in probe_checkpoint_paths:
        out_features = 20

        json_path = os.path.join(
            base_path, "class_mappings", "openmic_mapping.json")
        with open(json_path, "r") as f:
          class_idx_dict = json.load(f)  # this should be a dict

        self.class_map = {v: k for k, v in class_idx_dict.items()}

      elif "mtg_genre" in probe_checkpoint_paths:
        out_features = 87

        json_path = os.path.join(
            base_path, "class_mappings", "mtg_genre_mapping.json")
        with open(json_path, "r") as f:
          class_idx_dict = json.load(f)  # this should be a dict

        self.class_map = {v: k for k, v in class_idx_dict.items()}

      elif "mtg_instrument" in probe_checkpoint_paths:
        out_features = 40

        json_path = os.path.join(
            base_path, "class_mappings", "mtg_instrument_mapping.json")
        with open(json_path, "r") as f:
          class_idx_dict = json.load(f)  # this should be a dict

        self.class_map = {v: k for k, v in class_idx_dict.items()}

      elif "mtg_mood" in probe_checkpoint_paths:
        out_features = 56

        json_path = os.path.join(
            base_path, "class_mappings", "mtg_mood_mapping.json")
        with open(json_path, "r") as f:
          class_idx_dict = json.load(f)  # this should be a dict

        self.class_map = {v: k for k, v in class_idx_dict.items()}

      elif "mtg_top50" in probe_checkpoint_paths:
        out_features = 50

        json_path = os.path.join(
            base_path, "class_mappings", "mtg_top50_mapping.json")
        with open(json_path, "r") as f:
          class_idx_dict = json.load(f)  # this should be a dict

        self.class_map = {v: k for k, v in class_idx_dict.items()}

      elif "fsd50k" in probe_checkpoint_paths:
        out_features = 200
        csv_path = os.path.join(
            base_path, "class_mappings", "fsd50k_mapping.csv")
        df_class = pd.read_csv(csv_path, header=None, index_col=0)
        self.class_map = df_class[1].to_dict()

      elif "esc50" in probe_checkpoint_paths:
        out_features = 50

        json_path = os.path.join(
            base_path, "class_mappings", "esc50_mapping.json")
        with open(json_path, "r") as f:
          esc50_mapping = {int(k): v for k, v in json.load(f).items()}

        self.class_map = esc50_mapping

      self.probe = nn.Linear(in_features=3840, out_features=out_features)

  def preprocess(self, x):

    if x.ndim < 2:
      x = x.unsqueeze(dim=0)

    x = self.log_mel(x)

    x = (x - self.cfg.lms_mean) / self.cfg.lms_std

    return x

  def forward(self, x):

    x = self.preprocess(x)

    # TODO DELETE
    if x.ndim > 3:
      x = x.squeeze(dim=1)

    if self.inference_type == "precise":
      emb, layer_results = self.forward_precise(x)

    elif self.inference_type == "fast":
      emb, layer_results = self.forward_fast(x)

    if self.pull_time_dimension:
      emb = emb.mean(dim=1)
      layer_results = layer_results.mean(dim=2)

    return emb, layer_results

  def forward_fast(self, x):
    """Forward when we want to extract efficiently, better for finetuning,
    the forward is faster but adds a lot of padding sometimes compared to the
    precise forward.

    Parameters
    ----------
    x : torch.tensor
        of shape [bs, n_samples]

    Returns
    -------
    emb
        the embedding of the input audio of shape [bs, time, emb_dim]
    layer_results
        the output of each layer with shape [bs, n_layers, time, emb_dim]
    """

    bs, _, _ = x.shape

    patch_fbins = self.grid_size()[0]
    unit_frames = self.cfg.n_t
    embed_d = self.patch_embed.proj.out_channels

    n_chunk = (x.shape[-1] + unit_frames - 1) // unit_frames
    pad_frames = n_chunk*unit_frames - x.shape[-1]

    if pad_frames > 0:
      x = torch.nn.functional.pad(x, (0, pad_frames))

    x_full = rearrange(x, 'b f (n u) -> (b n) f u',
                       n=n_chunk, f=x.shape[-2], b=bs)

    _, layer_results_full = self.extract_features(
        x_full.unsqueeze(dim=1))

    layer_results_full = layer_results_full[..., 1:, :]

    if self.concat_freq:
      layer_results_full = rearrange(layer_results_full, 'b l (f t) d -> b l t (f d)',
                                     f=patch_fbins, d=embed_d)

      layer_results_full = rearrange(layer_results_full, '(b n) l t d -> b l (t n) d',
                                     b=bs, n=n_chunk, d=embed_d*patch_fbins)
    else:
      layer_results_full = rearrange(layer_results_full, '(b n) l t d -> b l (t n) d',
                                     b=bs, n=n_chunk, d=embed_d)

    emb = layer_results_full[:, -1]

    return emb, layer_results_full

  def forward_precise(self, x):
    """Forward that is precise but might be slow on big batches,
    this is the forward used for the results in the paper

    Parameters
    ----------
    x : torch.tensor
        of shape [bs, n_samples]

    Returns
    -------
    emb
        the embedding of the input audio of shape [bs, time, emb_dim]
    layer_results
        the output of each layer with shape [bs, n_layers, time, emb_dim]
    """

    patch_fbins = self.grid_size()[0]
    unit_frames = self.cfg.n_t
    patch_frames = self.patch_embed.patch_size[1]
    embed_d = self.patch_embed.proj.out_channels
    n_chunk = (x.shape[-1] + unit_frames - 1) // unit_frames
    pad_frames = (patch_frames - x.shape[-1] % patch_frames) % patch_frames
    if pad_frames > 0:
      x = torch.nn.functional.pad(x, (0, pad_frames))

    x = x.unsqueeze(dim=1)
    embeddings = []
    for i in range(n_chunk):
      emb, layer_results = self.extract_features(
          x[..., i*unit_frames:(i+1)*unit_frames])

      layer_results = layer_results[..., 1:, :]
      if self.concat_freq:
        layer_results = rearrange(layer_results, 'b n (f t) d -> b n t (f d)',
                                  f=patch_fbins, d=embed_d)

      elif self.as_class_head:
        layer_results = rearrange(layer_results, 'b n (f t) d -> b n t d f',
                                  f=patch_fbins, d=embed_d).mean(-1)

      embeddings.append(layer_results)
    layer_results = torch.cat(embeddings, axis=-2)

    emb = layer_results[:, -1]

    return emb, layer_results

  def probe_forward(self, x):
    x = self.preprocess(x)
    emb, layer_results = self.forward_precise(x)
    emb = emb.mean(dim=1)

    if self.as_class_head:
      emb = self.head_norm(emb.unsqueeze(-1)).squeeze(-1)
      probs = torch.nn.functional.softmax(self.head(emb), dim=-1)

    else:
      probs = torch.nn.functional.softmax(self.probe(emb), dim=-1)

    values, index = probs.topk(k=5, dim=-1)
    labels = [
        [f"{self.class_map[i]} ({v:.2f})" for i,
         v in zip(idx_row, val_row)]
        for idx_row, val_row in zip(index.cpu().numpy(), values.cpu().numpy())
    ]

    return probs, labels

  def extract_features(self, x):
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

    x = self.student_encoder.forward(x, return_layers=True)

    return x[:, -1, :, :], x  # Return embedding and layer results

  def grid_size(self):
    # This fails with timm 0.4.5 -> return self.patch_embed.grid_size
    # Workaround for avoid compatibility issue
    img_size = np.array(self.patch_embed.img_size)
    patch_size = np.array(self.patch_embed.patch_size)
    grid_size = img_size // patch_size
    return grid_size


def get_matpac(checkpoint_path,
               inference_type: str = "precise",  # or fast
               pull_time_dimension: bool = True,
               concat_freq: bool = True,
               as_class_head: bool = False,
               probe_checkpoint_paths: str = None):
  """Basic function to instantiate the inference model from the paper
  with the best checkpoint.

  Parameters
  ----------
  checkpoint_path : str
      The path to the checkpoint
  inference_type : str
      We have to type of inference. The main difference is the speed and its 
      precision. 
      With "precise" we use all the audio without adding some padding, but it 
      is slow on big batches as it rely on a loop.
      With "fast" we do some padding on the audio and we do not have a loop 
      to extract the features, it is faster but less precise.
  pull_time_dimension : bool
      Parameter to decide if we pull the time dimension during inference or not.
      If True we take the mean of the time dimension.
  concat_freq : bool
      True by default, it concatenate all embeddings along freq axis, if False
      the sequence is TxF. For AudioSet fine tuned models it is False by default.
  as_class_head : bool
      When loading a checkpoint of the model fine-tuned on AudioSet, set to True 
      to output the class prediction of AudioSet, set to False to access the 
      embeddings. 
  probe_checkpoint_paths : str
      Path for the weights of a probe trained on a downstream task. If it is not
      none it will instantiate the probe with the weights to give labels when calling
      "probe_forward".


  Returns
  -------
  _type_
      _description_
  """

  checkpoint = torch.load(checkpoint_path)

  if "as" in checkpoint_path:
    cfg = general_config(n_t=992)

    inference_type = "precise"
    print("Using precise inference by default with a CKPT finetuned on AudioSet")

    if "head" in checkpoint_path:
      concat_freq = False  # FALSE BY DEFAULT WHEN FT ON AS WITH HEAD
      pull_time_dimension = True  # TRUE BY DEFAULT WHEN FT ON AS WITH HEAD
      print("Using a CKPT finetuned on AudioSet: concat_freq=False and pull_time_dimension=True")

    else:
      as_class_head = False  # IF NO HEAD IN CHECKPOINT
      print("Using a CKPT finetuned on AudioSet but with encoder only")

  elif probe_checkpoint_paths is not None:
    inference_type = "precise"
    print("Using precise inference by default when using a downstream Probe")
    concat_freq = True  # True BY DEFAULT WHEN USING A DOWNSTREAM PROBE
    pull_time_dimension = True  # TRUE BY WHEN USING A DOWNSTREAM PROBE
    print("Using a Downstream probe: concat_freq=True and pull_time_dimension=True")
    cfg = general_config()

  else:
    cfg = general_config()
    # Forcing to false to avoid errors when the model is not fine tuned on AS
    as_class_head = False

  model = matpac_wrapper(inference_type=inference_type,
                         pull_time_dimension=pull_time_dimension,
                         as_class_head=as_class_head,
                         concat_freq=concat_freq,
                         cfg=cfg,
                         probe_checkpoint_paths=probe_checkpoint_paths)

  # Load the model state dict from the checkpoint into the model
  if probe_checkpoint_paths is not None:
    probe_checkpoint = torch.load(probe_checkpoint_paths)
    checkpoint = {**checkpoint, **probe_checkpoint}

  model.load_state_dict(checkpoint, strict=False)

  model.eval()

  return model


if __name__ == "__main__":

  dataset = "fsd50k"

  # ckpt_path = "/home/auquelennec/Bureau/ckpt/mcl_matpac/matpac_plus_as_48_1_map_enconly.pt"  # OK
  # ckpt_path = "/home/auquelennec/Bureau/ckpt/mcl_matpac/matpac_plus_music_6s_2048_enconly.pt"  # OK
  ckpt_path = "/home/auquelennec/Bureau/ckpt/mcl_matpac/matpac_plus_6s_2048_enconly.pt"  # OK
  # ckpt_path = "/home/auquelennec/Bureau/ckpt/mcl_matpac/matpac_plus_as_48_1_map_enc_and_head.pt"  # OK

  # model = get_matpac(checkpoint_path=ckpt_path,
  #                    as_class_head=True)

  model = get_matpac(checkpoint_path=ckpt_path,
                     probe_checkpoint_paths=f"/home/auquelennec/Bureau/ckpt/Weights_probes_matpac/matpac++_general_audio/{dataset}/{dataset}.pth"
                     #  f"/home/auquelennec/Bureau/ckpt/Weights_probes_matpac/matpac++_music/{dataset}/{dataset}.pth"
                     )

  import librosa
  wav, _ = librosa.load(
      # "/home/auquelennec/Bureau/audios_ds/openmic/017114_122880.ogg",
      "/home/auquelennec/Bureau/audios_ds/esc50/1-9887-B-49.wav",
      # "/home/auquelennec/Bureau/audios_ds/esc50/1-9887-A-49.wav",
      # "/home/auquelennec/Bureau/0/-1nilez17Dg_30.000_40.000.wav",
      # "/home/auquelennec/Bureau/0/0amHVSbkaX8_60.000_70.000.wav",
      mono=True, sr=16000
  )
  audio_real = torch.tensor(wav).unsqueeze(0)

  with torch.no_grad():
    prob, labels = model.probe_forward(audio_real)

  print(prob.shape)
  print(labels)

  # model = get_matpac(checkpoint_path=ckpt_path,
  #                    pull_time_dimension=False,
  #                    inference_type="fast",
  #                    as_class_head=True)

  # # audio = torch.rand((1, 160000))
  # audio_2 = torch.rand((2, 320000))

  # emb, layer_out = model(audio_2)
  # print(emb.shape)
  # print(layer_out.shape)

  # model = get_matpac(checkpoint_path=ckpt_path,
  #                    pull_time_dimension=True,
  #                    inference_type="precise",
  #                    as_class_head=True)

  # # audio = torch.rand((1, 160000))
  # import librosa
  # wav, _ = librosa.load(
  #     "/home/auquelennec/Bureau/0/0arN4pk83lA_30.000_40.000.wav", mono=True, sr=16000)
  # audio_real = torch.tensor(wav).unsqueeze(0)

  # with torch.no_grad():
  #   emb, layer_out = model(audio_real)

  # probs = emb.squeeze(0)
  # values, index = probs.topk(k=10)
  # print(', '.join([f'{model.as_class_map[i]} ({v*100:.1f}%)' for i,
  #       v in zip(index.numpy(), values.numpy())]))

  # print(emb.shape)
  # print(layer_out.shape)
