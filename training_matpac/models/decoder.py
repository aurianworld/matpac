"""Code for the transformer layers for the ViT encoder (teacher and student)"""
# Modified from https://github.com/nttcslab/m2d

import torch
import torch.nn as nn
from dataclasses import dataclass
from functools import partial
from omegaconf import II

from timm.models.vision_transformer import Block


@dataclass
class decoder_config:
  embed_dim: int = II("model.encoder.embed_dim")
  decoder_embed_dim: int = 512
  decoder_depth: int = 8
  mlp_ratio: int = 4
  decoder_num_heads: int = 16


class decoder(nn.Module):
  """Decoder, projections and transformer layers.
  """

  def __init__(self,
               cfg: decoder_config,
               num_patches):
    """
      Parameters
      ----------
      cfg : decoder_config
          dataclass with all the parameters for the transformer layers and the 
          projection layers.
    """
    super().__init__()
    self.decoder_embed = nn.Linear(
        cfg.embed_dim, cfg.decoder_embed_dim, bias=True)

    self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder_embed_dim))

    self.decoder_pos_embed = nn.Parameter(torch.zeros(
        1, num_patches + 1, cfg.decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

    self.decoder_blocks = nn.ModuleList([
        Block(cfg.decoder_embed_dim, cfg.decoder_num_heads,
              cfg.mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        for i in range(cfg.decoder_depth)])

    self.decoder_norm = nn.LayerNorm(cfg.decoder_embed_dim, eps=1e-6)

    self.decoder_pred = nn.Linear(
        cfg.decoder_embed_dim, cfg.embed_dim, bias=True)

  def forward(self, x,
              return_layers=False):
    """Forward of a sequence through the transformer layers

    Parameters
    ----------
    x : torch.tensor
        the sequence to pass through the layers
    return_layers : bool, optional
        If true returns the output of each layers, by default False

    Returns
    -------
    torch.tensor
        Either the output of the last layer, or the stacked output of each
        layer.
    """

    x = self.decoder_embed(x)

    if x.shape[1] <= self.decoder_pos_embed.shape[1]:
      x = x + self.decoder_pos_embed[:, :x.shape[1], :]
    else:
      x = x + self.decoder_pos_embed

    layers = []
    for blk in self.blocks:
      x = blk(x)
      if return_layers:
        layers.append(x.unsqueeze(dim=1))
    x = self.norm(x)
    if return_layers:
      layers[-1] = x.unsqueeze(dim=1)
      return torch.cat(layers, dim=1)
    else:
      return x
