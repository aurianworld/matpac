import numpy as np
import torch
import torch.nn as nn


def expand_size(sz):
  if isinstance(sz, int):
    return [sz, sz]
  return sz


class PatchEmbed(nn.Module):
  """ 2D Image to Patch Embedding -- borrowed from https://pypi.org/project/timm/0.4.12/
  """

  def __init__(self, img_size=[80, 304], patch_size=16, in_chans=1, embed_dim=768, norm_layer=None, flatten=True):
    super().__init__()
    img_size = expand_size(img_size)
    patch_size = expand_size(patch_size)
    self.img_size = img_size
    self.patch_size = patch_size
    self.grid_size = (img_size[0] // patch_size[0],
                      img_size[1] // patch_size[1])
    self.num_patches = self.grid_size[0] * self.grid_size[1]
    self.flatten = flatten

    self.proj = nn.Conv2d(in_chans, embed_dim,
                          kernel_size=patch_size, stride=patch_size)
    self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

  def forward(self, x):
    x = self.proj(x)
    if self.flatten:
      x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
    x = self.norm(x)
    return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
  """
  grid_size: int of the grid height and width
  return:
  pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
  """
  gH, gW = grid_size
  grid_h = np.arange(gH, dtype=np.float32)
  grid_w = np.arange(gW, dtype=np.float32)
  grid = np.meshgrid(grid_w, grid_h)  # here w goes first
  grid = np.stack(grid, axis=0)

  grid = grid.reshape([2, 1, gH, gW])
  pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
  if cls_token:
    pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
  return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
  assert embed_dim % 2 == 0

  # use half of dimensions to encode grid_h
  emb_h = get_1d_sincos_pos_embed_from_grid(
      embed_dim // 2, grid[0])  # (H*W, D/2)
  emb_w = get_1d_sincos_pos_embed_from_grid(
      embed_dim // 2, grid[1])  # (H*W, D/2)

  emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
  return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
  """
  embed_dim: output dimension for each position
  pos: a list of positions to be encoded: size (M,)
  out: (M, D)
  """
  assert embed_dim % 2 == 0
  omega = np.arange(embed_dim // 2, dtype=float)
  omega /= embed_dim / 2.
  omega = 1. / 10000**omega  # (D/2,)

  pos = pos.reshape(-1)  # (M,)
  out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

  emb_sin = np.sin(out)  # (M, D/2)
  emb_cos = np.cos(out)  # (M, D/2)

  emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
  return emb
