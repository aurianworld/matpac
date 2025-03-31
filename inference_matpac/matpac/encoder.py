"""Code for the transformer layers for the ViT encoder (teacher and student)"""
# modified from https://github.com/nttcslab/m2d

import torch
import torch.nn as nn
from dataclasses import dataclass
from functools import partial

from timm.models.vision_transformer import Block


@dataclass
class encoder_layers_config:
  embed_dim: int = 768
  depth: int = 12
  num_heads: int = 12
  mlp_ratio: int = 4


class encoder_layers(nn.Module):
  """Vision transformer encoder layers.
  """

  def __init__(self,
               cfg: encoder_layers_config):
    """
      Parameters
      ----------
      cfg : encoder_layers_config
          dataclass with all the parameters for the transformer layers and the 
          layer norm.
    """
    super().__init__()
    self.blocks = nn.ModuleList([
        Block(cfg.embed_dim,
              cfg.num_heads,
              cfg.mlp_ratio,
              qkv_bias=True,
              norm_layer=partial(nn.LayerNorm, eps=1e-6))
        for i in range(cfg.depth)])
    self.norm = nn.LayerNorm(cfg.embed_dim, eps=1e-6)

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

    layers = []
    for blk in self.blocks:
      x = blk(x)
      if return_layers:
        layers.append(x.unsqueeze(dim=1))
    x = self.norm(x)
    if return_layers:
      # replace the last feature with the normalized one.
      layers[-1] = x.unsqueeze(dim=1)
      return torch.cat(layers, dim=1)
    else:
      return x
