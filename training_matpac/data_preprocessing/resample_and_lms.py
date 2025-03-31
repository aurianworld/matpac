import os
import concurrent.futures

import numpy as np
import soundfile as sf

import torch
import torchaudio
import argparse
import time


class logMelSpectrogram(torch.nn.Module):
  r"""Create MelSpectrogram for a raw audio signal. Wrapping of torchaudio class, but
  with parameters from librosa and time input in seconds for convenience.

  .. devices:: CPU CUDA

  .. properties:: Autograd TorchScript

  This is a composition of :py:func:`torchaudio.transforms.Spectrogram` and
  and :py:func:`torchaudio.transforms.MelScale`.

  Sources
      * https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
      * https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html
      * http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

  Args:
      sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
      n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
      win_length (float or None, optional): Window size. (Default: ``n_fft``)
      hop_length (float or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
      f_min (float, optional): Minimum frequency. (Default: ``0.``)
      f_max (float or None, optional): Maximum frequency. (Default: ``None``)
      pad (int, optional): Two sided padding of signal. (Default: ``0``)
      n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
      window_fn (Callable[..., Tensor], optional): A function to create a window tensor
          that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
      power (float, optional): Exponent for the magnitude spectrogram,
          (must be > 0) e.g., 1 for energy, 2 for power, etc. (Default: ``2``)
      normalized (bool, optional): Whether to normalize by magnitude after stft. (Default: ``False``)
      wkwargs (Dict[..., ...] or None, optional): Arguments for window function. (Default: ``None``)
      center (bool, optional): whether to pad :attr:`waveform` on both sides so
          that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
          (Default: ``True``)
      pad_mode (string, optional): controls the padding method used when
          :attr:`center` is ``True``. (Default: ``"reflect"``)
      onesided: Deprecated and unused.
      norm (str or None, optional): If "slaney", divide the triangular mel weights by the width of the mel band
          (area normalization). (Default: ``None``)
      mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

  Example
      >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
      >>> transform = transforms.MelSpectrogram(sample_rate)
      >>> mel_specgram = transform(waveform)  # (channel, n_mels, time)

  See also:
      :py:func:`torchaudio.functional.melscale_fbanks` - The function used to
      generate the filter banks.
  """

  def __init__(self,
               sample_rate=16000,
               n_fft=None,
               win_length=0.025,
               hop_length=0.01,
               f_min=0.0,
               f_max=None,
               log_offset=0.001,
               pad=0,
               n_mels=128,
               window_fn=torch.hann_window,
               power=2.0,
               normalized=False,
               wkwargs=None,
               center=False,
               pad_mode="reflect",
               onesided=None,
               norm="slaney",
               mel_scale="slaney",
               ) -> None:
    super(logMelSpectrogram, self).__init__()

    if f_max is None:
      f_max = sample_rate // 2

    win_length = int(np.round(sample_rate * win_length))

    if n_fft is None:
      n_fft = win_length

    if hop_length is None:
      hop_length = win_length // 2
    else:
      hop_length = int(np.round(sample_rate * hop_length))

    self.MelSpectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        pad=pad,
        n_mels=n_mels,
        window_fn=window_fn,
        power=power,
        normalized=normalized,
        wkwargs=wkwargs,
        center=center,
        pad_mode=pad_mode,
        onesided=onesided,
        norm=norm,
        mel_scale=mel_scale)

    self.log_offset = log_offset

  def forward(self, waveform):
    mel_specgram = self.MelSpectrogram(waveform)
    log_melspecgram = torch.log(mel_specgram + self.log_offset)
    return log_melspecgram


class slice_audio(torch.nn.Module):
  """Prepare audio tensor to be sliced into segments of length window_length

  Parameters
  ----------
  sr : int
      Sampling rate of the audio
  window_length : int
      Length in seconds of the slices
  step_size : float
      Length in seconds of the step between two windows."""

  def __init__(self, sr=16000,
               window_length=1,
               step_size=1,
               add_last=False) -> None:
    super(slice_audio, self).__init__()

    self.sr = sr
    self.window_length = window_length
    self.step_size = step_size
    self.add_last = add_last

  def forward(self, waveform):

    n_samples = waveform.shape[-1]
    segment_size = int(self.window_length * self.sr)
    step_size = int(self.step_size * self.sr)
    n_segments = int(1 + (n_samples - segment_size) // step_size)

    if self.add_last:
      audio_segments = torch.zeros((n_segments+1, segment_size))
    else:
      audio_segments = torch.zeros((n_segments, segment_size))
    device = waveform.device
    audio_segments = audio_segments.to(device=device)

    i = 0
    for i in range(n_segments):
      start = i * step_size
      end = start + segment_size
      audio_segments[i] = waveform[:, start:end]

    if self.add_last:

      # If waveform shorter than the slice len
      if i == 0 and n_segments == 0:
        len_last = waveform.shape[-1]
        audio_segments[0][:len_last] = waveform
      else:
        len_last = waveform[:, end:].shape[-1]
        audio_segments[i+1][:len_last] = waveform[:, end:]

    return audio_segments


def to_log_mel(input_audio_path,
               output_audio_path,
               sample_rate,
               logmel,
               device):
  """Function to resample, mono and compute logmel of a track from it's input
  path and save it in npy format.

  Parameters
  ----------
  input_audio_path : str
    Path to the audio to process
  output_audio_path : str
    Path of where to save the processed audio
  sample_rate : int
    Sample rate to be used as target
  logmel : logMelSpectrogram object
    Object to compute logmel spectrogram
  device : torch.device
    CPU or CUDA device
  """

  # Loading audio
  try:
    audio_tensor, sr = sf.read(
        input_audio_path,
        dtype="float32",
        always_2d=True,
    )

  except BaseException:
    print("corrupted:", input_audio_path)
    return

  # To Mono
  audio_tensor = torch.from_numpy(audio_tensor)  # (n_samples, n_channels)
  audio_tensor = torch.mean(
      audio_tensor,
      dim=-1,
      keepdim=False)  # (n_samples,)

  if audio_tensor.shape[-1] < 1200:  # Less than a window of fft for most sr
    print("No samples:", input_audio_path)
    return

  else:
    if sr != sample_rate:
      # Resample audio to target sample rate, shape (n_samples,)
      audio_tensor = torchaudio.functional.resample(
          waveform=audio_tensor,
          orig_freq=sr,
          new_freq=sample_rate,
          lowpass_filter_width=64,
          rolloff=0.9475937167399596,
          resampling_method="sinc_interp_kaiser",
          beta=14.769656459379492,
      )

  audio_tensor = audio_tensor.to(device)
  logmel_func = logmel.to(device)

  # Compute logmel, shape (n_mel, n_frames)
  logmel_tensor = logmel_func(audio_tensor)

  # Save logmel
  np.save(output_audio_path,
          logmel_tensor.cpu().numpy())

  # Free memory
  del audio_tensor
  del logmel_tensor
  torch.cuda.empty_cache()


def process_folder(input_folder,
                   output_folder,
                   sample_rate,
                   n_mel,
                   num_threads,
                   replace=False):
  """Function to process a dataset folder and create a version where every
  audio is resampled to the desire sampling rate and mono.

  Parameters
  ----------
  input_folder : str
    Path of the input dataset
  output_folder : str
    Path of the output of the processed dataset
  target_sample_rate : int
    Target sample rate for the processed dataset
  num_threads : int
    Number of thread to be used in the multiprocessing
  replace : bool
    bool to replace existing output files or not
  """

  logmel = logMelSpectrogram(
      sample_rate=sample_rate,
      win_length=0.025,
      hop_length=0.01,
      f_min=50,
      f_max=sample_rate // 2,
      log_offset=torch.finfo().eps,
      n_mels=n_mel,
      center=False,
  )

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  for root, _, files in os.walk(input_folder):
    print("\ntreating: ", root)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:

      for file in files:
        input_path = os.path.join(root, file)

        output_path = os.path.join(
            output_folder,
            os.path.relpath(input_path, input_folder),
        )

        if not os.path.exists(os.path.dirname(output_path)):
          os.makedirs(os.path.dirname(output_path))

        if input_path.lower().endswith((".wav", ".mp3")):
          output_path, _ = os.path.splitext(output_path)
          output_path = output_path + ".npy"

          if os.path.isfile(output_path) and not replace:
            pass

          else:
            executor.submit(
                to_log_mel,
                input_path,
                output_path,
                sample_rate,
                logmel,
                device,
            )


if __name__ == "__main__":

  parser = argparse.ArgumentParser(
      description="Copy and resample audio files.")
  parser.add_argument("--input_folder", help="Input folder path")
  parser.add_argument("--output_folder", help="Output folder path")
  parser.add_argument("--sample_rate", type=int, default=16000,
                      help="Target sampling rate (default: 16000)")
  parser.add_argument("--n_mel", type=int, default=80,
                      help="number of n_mel")
  parser.add_argument("--num_threads", type=int, default=None,
                      help="Number of threads to use in the function")

  # Note:
  # If replace is False and the output file already exists, the function
  # will not process the file.
  # Else, if replace is True, the function will process the file and replace
  # the existing file if it exists.
  parser.add_argument(
      "--replace", type=bool, default=True,
      help="To replace existing files in the folders or not (default: False).")

  args = parser.parse_args()

  slurm = "SLURM_JOB_ID" in os.environ
  if slurm:
    num_threads = int(os.environ["SLURM_CPUS_ON_NODE"])
  elif args.num_threads is not None:
    num_threads = args.num_threads
  else:
    num_threads = os.cpu_count()

  print("Number of cpu {}".format(num_threads))

  start_time = time.time()

  process_folder(
      input_folder=args.input_folder,
      output_folder=args.output_folder,
      sample_rate=args.sample_rate,
      n_mel=args.n_mel,
      num_threads=num_threads,
      replace=args.replace,
  )

  # Record time
  elapsed_time = time.time() - start_time
  print(f"Done! Files have been resampled transformed to logmel in {elapsed_time:.4f}s")  # nopep8
