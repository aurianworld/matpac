#!/usr/bin/env python3

# Modified code from Facebook's Fairseq

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: goes through a folder and create csv file listing all audios path.
"""

import argparse
import glob
import os
from tqdm import tqdm
import numpy as np

import soundfile


def main(args):
  """Create a tsv file containing the path of all the file to process in a fairseq training

  Parameters
  ----------
  args : dict
      see parser documentation
  """
  if args.ext == "wav":
    min_frame = 16000
  else:
    min_frame = 100

  if not os.path.exists(args.dest):
    os.makedirs(args.dest)

  dir_path = os.path.realpath(args.root)
  search_path = os.path.join(dir_path, f"**/*.{args.ext}")

  train_f = open(os.path.join(args.dest, "train.tsv"), "w")

  print(dir_path, file=train_f)

  for fname in tqdm(glob.iglob(search_path, recursive=True)):
    file_path = os.path.realpath(fname)

    if args.ext == "wav":
      frames = soundfile.info(fname).frames
    elif args.ext == "npy":
      frames = np.load(fname).shape[-1]
    else:
      return print("Extension not handled")

    if frames > min_frame:
      print(f"{os.path.relpath(file_path, dir_path)}\t{frames}", file=train_f)

  if train_f:
    train_f.close()


def get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "root", metavar="DIR",
      help="root directory containing wav or npy files to index")
  parser.add_argument(
      "--dest", default=".", type=str, metavar="DIR", help="output directory"
  )
  parser.add_argument(
      "--ext",
      default="wav",
      type=str,
      metavar="EXT",
      help="extension to look for")
  parser.add_argument(
      "--path-must-contain",
      default=None,
      type=str,
      metavar="FRAG",
      help="if set, path must contain this substring for a file to be included in the manifest",
  )
  return parser


if __name__ == "__main__":
  parser = get_parser()
  args = parser.parse_args()
  main(args)
