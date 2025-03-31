import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from einops import rearrange

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
               cfg: general_config = general_config(),
               ):
    super(matpac_wrapper, self).__init__()

    # Setting the parameters
    self.cfg = cfg
    self.inference_type = inference_type
    self.pull_time_dimension = pull_time_dimension

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

    layer_results_full = rearrange(layer_results_full, 'b l (f t) d -> b l t (f d)',
                                   f=patch_fbins, d=embed_d)

    layer_results_full = rearrange(layer_results_full, '(b n) l t d -> b l (t n) d',
                                   b=bs, n=n_chunk, d=embed_d*patch_fbins)

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
      layer_results = rearrange(layer_results, 'b n (f t) d -> b n t (f d)',
                                f=patch_fbins, d=embed_d)
      embeddings.append(layer_results)
    layer_results = torch.cat(embeddings, axis=-2)

    layer_results = layer_results
    emb = layer_results[:, -1]

    return emb, layer_results

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
               pull_time_dimension: bool = True):
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


  Returns
  -------
  _type_
      _description_
  """

  model = matpac_wrapper(inference_type=inference_type,
                         pull_time_dimension=pull_time_dimension,
                         )

  checkpoint = torch.load(checkpoint_path)

  # Load the model state dict from the checkpoint into the model
  model.load_state_dict(checkpoint, strict=False)

  model.eval()

  return model


if __name__ == "__main__":

  model = get_matpac(checkpoint_path=ckpt_path, pull_time_dimension=False)

  audio = torch.rand((1, 160000))
  audio_2 = torch.rand((2, 320000))

  print(model(audio))
