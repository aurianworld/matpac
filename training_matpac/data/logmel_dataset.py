# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Optional
import numpy as np

import torch
from fairseq.data.fairseq_dataset import FairseqDataset
from fairseq.data.audio.audio_utils import (
    parse_path,
)

logger = logging.getLogger(__name__)


def load_audio(manifest_path, max_keep, min_keep):
  n_long, n_short = 0, 0
  names, inds, sizes = [], [], []
  with open(manifest_path) as f:
    root = f.readline().strip()
    for ind, line in enumerate(f):
      items = line.strip().split("\t")
      assert len(items) == 2, line
      sz = int(items[1])
      if min_keep is not None and sz < min_keep:
        n_short += 1
      elif max_keep is not None and sz > max_keep:
        n_long += 1
      else:
        names.append(items[0])
        inds.append(ind)
        sizes.append(sz)
  tot = ind + 1
  logger.info(
      (
          f"max_keep={max_keep}, min_keep={min_keep}, "
          f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
          f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
      )
  )
  return root, names, inds, tot, sizes


class LogmelDataset(FairseqDataset):
  def __init__(
      self,
      manifest_path: str,
      sample_rate: float,
      max_keep_sample_size: Optional[int] = None,
      min_keep_sample_size: Optional[int] = None,
      max_sample_size: Optional[int] = None,

      # Other Options
      shuffle: bool = True,
      pad_audio: bool = False,
      random_crop: bool = False,
      hop_length: float = 0.01,
      n_mels: int = 80,
      patch_size: int = 16,
      lms_mean: float = -7.056,
      lms_std: float = 4.193,
  ):
    self.audio_root, self.audio_names, inds, tot, self.sizes = load_audio(
        manifest_path, max_keep_sample_size, min_keep_sample_size
    )
    self.sample_rate = sample_rate
    self.logmel_rate = int(1/hop_length)
    self.n_mels = n_mels
    self.n_freq_patch = n_mels // patch_size
    self.patch_size = patch_size
    self.norm_stats = [lms_mean, lms_std]
    self.shuffle = shuffle
    self.random_crop = random_crop

    self.max_mel_size = max_sample_size
    self.max_sample_size = max_sample_size

    self.pad_audio = pad_audio

    logger.info(
        f"pad_audio={pad_audio}, random_crop={random_crop}, "
        f"max_sample_size={max_sample_size}"
    )

  def get_lms(self, index):

    lms_path = os.path.join(self.audio_root, self.audio_names[index])
    _path, slice_ptr = parse_path(lms_path)
    if len(slice_ptr) == 0:
      lms = np.load(_path)
    lms = torch.from_numpy(lms).float()

    lms = (lms - self.norm_stats[0]) / self.norm_stats[1]

    return lms

  def __getitem__(self, index):

    logmel = self.get_lms(index)

    return {"id": index, "source": logmel}

  def __len__(self):
    return len(self.sizes)

  def crop_to_max_size(self, logmel, target_size):
    size = logmel.shape[-1]
    diff = size - target_size
    if diff <= 0:
      return logmel, 0

    start, end = 0, target_size
    if self.random_crop:
      start = np.random.randint(0, diff + 1)
      # start = np.random.randint(0, diff)
      end = size - diff + start
    return logmel[:, start:end], start

  def collater(self, samples):
    # target = max(sizes) -> random_crop not used
    # target = max_sample_size -> random_crop used for long
    samples = [s for s in samples if s["source"] is not None]
    if len(samples) == 0:
      return {}

    logmels = [s["source"] for s in samples]
    logmels_sizes = [s.shape[-1] for s in logmels]

    if self.pad_audio:
      logmels_size = min(max(logmels_sizes), self.max_mel_size)
    else:
      logmels_size = min(min(logmels_sizes), self.max_mel_size)
    collated_logmels, _ = self.collater_logmel(
        logmels, logmels_size
    )

    net_input = {"source": collated_logmels}

    batch = {
        "id": torch.LongTensor([s["id"] for s in samples]),
    }

    batch["net_input"] = net_input

    return batch

  def collater_logmel(self, logmels, logmel_size):
    collated_logmels = logmels[0].new_zeros(
        len(logmels), self.n_mels, logmel_size)

    logmel_starts = [0 for _ in logmels]
    for i, logmel in enumerate(logmels):
      diff = logmel.shape[-1] - logmel_size
      if diff == 0:
        collated_logmels[i] = logmel
      elif diff < 0:
        assert self.pad_audio
        collated_logmels[i] = torch.hstack(
            [logmel, logmel.new_full((logmel.shape[0], -diff,), 0.0)])
      else:
        collated_logmels[i], logmel_starts[i] = self.crop_to_max_size(
            logmel, logmel_size
        )
    return collated_logmels, logmel_starts

  def num_tokens(self, index):
    return self.size(index)

  def size(self, index):
    if self.pad_audio:
      return self.sizes[index]
    return min(self.sizes[index], self.max_sample_size)

  def ordered_indices(self):
    if self.shuffle:
      order = [np.random.permutation(len(self))]
    else:
      order = [np.arange(len(self))]

    order.append(self.sizes)
    return np.lexsort(order)[::-1]
