# Modified from https://github.com/facebookresearch/dino

from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm


@dataclass
class Cls_Head_Config:
  student_temp: float = field(default=0.1,
                              metadata={
                                  "help": "temperature for student softmax"})

  warmup_teacher_temp: float = field(default=0.04,
                                     metadata={
                                         "help": "temperature for teacher softmax"})
  teacher_temp: float = field(default=0.07,
                              metadata={
                                  "help": "temperature for teacher softmax"})
  warmup_n_steps: int = field(default=20000,
                              metadata={"help": "Number of steps for the warmup \
                                of the teacher temp"})
  center_momentum: float = field(default=0.9,
                                 metadata={
                                     "help": "Centering momentum when updating"})
  head_dim_out: int = field(default=2048,
                            metadata={"help": "Output Dimension of the proj \
                            head for online loss"})
  hidden_dim: int = 2048
  ema_decay: float = field(default=0.998, metadata={
                           "help": "initial ema decay rate"})
  ema_end_decay: float = field(
      default=1, metadata={"help": "final ema decay rate"})


class Cls_Head(nn.Module):
  def __init__(
      self,
      in_dim,
      out_dim=4096,
      nlayers=3,
      hidden_dim=2048,
      bottleneck_dim=256,
      mlp_bias=True,
  ):
    super().__init__()
    nlayers = max(nlayers, 1)
    self.mlp = _build_mlp(nlayers, in_dim, bottleneck_dim,
                          hidden_dim=hidden_dim, bias=mlp_bias)
    self.apply(self._init_weights)
    self.last_layer = weight_norm(
        nn.Linear(bottleneck_dim, out_dim, bias=False))
    self.last_layer.weight_g.data.fill_(1)

  def _init_weights(self, m):
    if isinstance(m, nn.Linear):
      trunc_normal_(m.weight, std=0.02)
      if isinstance(m, nn.Linear) and m.bias is not None:
        nn.init.constant_(m.bias, 0)

  def forward(self, x):
    x = self.mlp(x)
    eps = 1e-6 if x.dtype == torch.float16 else 1e-12
    x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
    x = self.last_layer(x)
    return x


def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, bias=True):
  if nlayers == 1:
    return nn.Linear(in_dim, bottleneck_dim, bias=bias)
  else:
    layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
    layers.append(nn.GELU())
    for _ in range(nlayers - 2):
      layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
      layers.append(nn.GELU())
    layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
    return nn.Sequential(*layers)


class Cls_Loss(nn.Module):
  def __init__(
      self,
      out_dim,
      student_temp=0.1,
      center_momentum=0.9,
  ):
    super().__init__()
    self.student_temp = student_temp
    self.center_momentum = center_momentum
    self.register_buffer("center", torch.zeros(1, 1, out_dim))
    self.updated = True
    self.reduce_handle = None
    self.len_teacher_output = None
    self.async_batch_center = None

  @torch.no_grad()
  def softmax_center_teacher(self, teacher_output, teacher_temp):
    self.apply_center_update()
    # teacher centering and sharpening
    return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

  def forward(self, student_output, teacher_out_softmaxed_centered, result):
    """
    Cross-entropy between softmax outputs of the teacher and student networks.
    """
    # We go from (B, N, D) to (B*N, D) and apply temperature to the student
    temp_student = student_output / self.student_temp
    temp_student = temp_student.view((-1, temp_student.shape[-1]))
    t = teacher_out_softmaxed_centered.view(
        (-1, teacher_out_softmaxed_centered.shape[-1]))

    loss = torch.sum(
        t * F.log_softmax(temp_student, dim=-1),
        dim=-1
    )

    temp_student = torch.softmax(temp_student, dim=-1)
    result["var_p_hat_m"] = self.compute_var(temp_student) * 1000
    result["var_p_m"] = self.compute_var(t) * 1000
    result = self.compute_acc(temp_student, t, result)

    return -loss, result

  def compute_acc(self, pred, target, result):

    result["cls_perplexity"] = torch.exp(
        (-torch.sum((target.mean(dim=0) *
                     torch.log(target.mean(dim=0)+1e-7)), dim=-1)))

    idx_targ = torch.argmax(target, dim=-1)
    idx_pred = torch.argmax(pred, dim=-1)

    n_diff_target = idx_targ.float().unique().numel()
    result["n_diff_target"] = n_diff_target

    correct = (idx_targ == idx_pred).long().sum().item()
    count = idx_targ.numel()

    result["var_idx_targ"] = self.compute_var(idx_targ.unsqueeze(dim=1))
    result["var_idx_pred"] = self.compute_var(idx_pred.unsqueeze(dim=1))
    result["correct"] = correct
    result["count"] = count

    return result

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

  @torch.no_grad()
  def update_center(self, teacher_output):
    self.reduce_center_update(teacher_output)

  @torch.no_grad()
  def reduce_center_update(self, teacher_output):
    self.updated = False
    self.len_teacher_output = teacher_output.shape[0] * teacher_output.shape[1]
    # Sum over batch dimension
    self.async_batch_center = torch.sum(
        teacher_output, dim=0, keepdim=True)
    # Sum over the number of patch/class tokens
    self.async_batch_center = torch.sum(
        self.async_batch_center, dim=1, keepdim=True)

    if dist.is_initialized():
      self.reduce_handle = dist.all_reduce(
          self.async_batch_center, async_op=True)

  @torch.no_grad()
  def apply_center_update(self):
    if self.updated is False:
      world_size = dist.get_world_size() if dist.is_initialized() else 1

      if self.reduce_handle is not None:
        self.reduce_handle.wait()

      _t = self.async_batch_center / (self.len_teacher_output * world_size)

      self.center = self.center * self.center_momentum + \
          _t * (1 - self.center_momentum)

      self.updated = True
