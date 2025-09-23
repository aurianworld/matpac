# Code of the MCL heads that follows the decoder
import torch
import torch.nn as nn
from dataclasses import dataclass
from omegaconf import II


@dataclass
class mcl_config:

  # Head params
  embed_dim: int = II("model.encoder.embed_dim")
  decoder_embed_dim: int = II("model.decoder.decoder_embed_dim")
  n_head: int = 5

  # if scoring
  score: bool = False

  # Temperature param
  temperature_start: float = 1
  temperature_end: float = 0.0001
  decay_rate: float = 0.99997


class mcl_heads(nn.Module):
  """Decoder, projections and transformer layers.
  """

  def __init__(self,
               cfg: mcl_config,
               ):
    """
      Parameters
      ----------
      cfg : mcl_config
          dataclass with all the parameters for the mcl heads.
    """
    super().__init__()
    self.mcl_heads = nn.ModuleList([nn.Linear(cfg.decoder_embed_dim,
                                              cfg.embed_dim)
                                    for i in range(cfg.n_head)])

    if cfg.score:
      self.scoring_head = nn.Linear(cfg.decoder_embed_dim,
                                    cfg.n_head)
    else:
      self.scoring_head = None

  def forward(self, x):
    """Forward the sequence from the decoder into each head and forward a 
    score for which head might be the best to use

    Parameters
    ----------
    x : torch.Tensor [batch_size, n_token, decoder_embed_dim]
        the output of the decoder transformer layers

    Returns
    -------
    mcl_preds : torch.Tensor [batch_size, n_heads, n_token, embed_dim]
      The prediction of each head for the reconstruction of masked patches in
      the latent space.
    head_scores : torch.Tensor [[batch_size, n_heads]
      The score for each head, trying to predict which head we should use for 
      backward.
    """

    if self.scoring_head is not None:
      head_scores = self.scoring_head(x)
      head_scores = torch.sigmoid(head_scores)
    else:
      head_scores = None

    mcl_preds = []
    for head in self.mcl_heads:
      pred = head(x)
      mcl_preds.append(pred)

    mcl_preds = torch.stack(mcl_preds, dim=2)

    return mcl_preds, head_scores
