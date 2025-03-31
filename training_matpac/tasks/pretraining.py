# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
from typing import Optional, Tuple

import numpy as np

from dataclasses import dataclass, field
from ..data.logmel_dataset import LogmelDataset
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING

logger = logging.getLogger(__name__)


@dataclass
class PretrainingConfig(FairseqDataclass):
  data: str = field(default=MISSING, metadata={
                    "help": "path to data directory"})

  sample_rate: int = field(
      default=16_000,
      metadata={
          "help": "target sample rate. audio files will be up/down "
          "sampled to this rate"
      },
  )
  n_mels: int = field(
      default=80,
      metadata={
          "help": "number of mel bins"
          "n mel bins"
      },
  )
  patch_size: int = field(
      default=16, metadata={"help": "patch size"}
  )
  lms_mean: float = field(
      default=-7.056,
      metadata="The mean computed on all the logmelspec of the dataset"
  )
  lms_std: float = field(
      default=4.193,
      metadata="The std computed on all the logmelspec of the dataset"
  )
  enable_padding: bool = field(
      default=False,
      metadata={"help": "pad shorter samples instead of cropping"},
  )
  max_keep_size: Optional[int] = field(
      default=None,
      metadata={"help": "exclude sample longer than this"},
  )
  max_sample_size: Optional[int] = field(
      default=None,
      metadata={"help": "max sample size to crop to for batching"},
  )
  min_sample_size: Optional[int] = field(
      default=None,
      metadata={"help": "min sample size to crop to for batching"},
  )
  random_crop: Optional[bool] = field(
      default=True,
      metadata={"help": "always crop from the beginning if false"},
  )
  pad_audio: Optional[bool] = field(
      default=False,
      metadata={"help": "pad audio to the longest one in the batch if true"},
  )
  rebuild_batches: bool = True


@register_task("logmel_pretraining", dataclass=PretrainingConfig)
class LogmelPretrainingTask(FairseqTask):

  cfg: PretrainingConfig

  def __init__(
      self,
      cfg: PretrainingConfig,
  ) -> None:
    super().__init__(cfg)

    logger.info(f"current directory is {os.getcwd()}")
    logger.info(f"PretrainingTask Config {cfg}")

    self.cfg = cfg

  @classmethod
  def setup_task(
      cls, cfg: PretrainingConfig, **kwargs
  ) -> "PretrainingTask":
    return cls(cfg)

  def load_dataset(self, split: str, **kwargs) -> None:
    manifest = f"{self.cfg.data}/{split}.tsv"

    self.datasets[split] = LogmelDataset(
        manifest,
        sample_rate=self.cfg.sample_rate,
        max_keep_sample_size=self.cfg.max_keep_size,
        min_keep_sample_size=self.cfg.min_sample_size,
        max_sample_size=self.cfg.max_sample_size,
        pad_audio=self.cfg.pad_audio,
        random_crop=self.cfg.random_crop,
        n_mels=self.cfg.n_mels,
        patch_size=self.cfg.patch_size,
        lms_mean=self.cfg.lms_mean,
        lms_std=self.cfg.lms_std,
    )

  def max_positions(self) -> Tuple[int, int]:
    return (sys.maxsize, sys.maxsize)

  def filter_indices_by_size(self, indices: np.array, *args, **kwargs) -> np.array:
    return indices
