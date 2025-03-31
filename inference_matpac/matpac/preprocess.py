
import numpy as np

import torch
import torchaudio


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
