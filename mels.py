"""Main script of wave-to-mel preprocessing."""


import argparse
from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.functional import resample
import torchaudio.transforms as transforms


class LogMelSpectrogram(torch.nn.Module):
    """Waveform-to-LogMelSpec, hop 10msec, 128-dim, same/center padding."""
    def __init__(self):
        super().__init__()
        self.melspctrogram = transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=160,
            center=False,
            power=1.0,
            norm="slaney",
            onesided=True,
            n_mels=128,
            mel_scale="slaney",
        )

    def forward(self, wav):
        # `win_length - hop_length`, keep frame-block correspondence (~= "same" padding)
        len_pad_same = 1024 - 160
        # pad_L == pad_R, so centered padding
        padding = len_pad_same // 2
        wav = F.pad(wav, (padding, padding), "reflect")

        # wave-to-logmelspec
        mel = self.melspctrogram(wav)
        logmel = torch.log(torch.clamp(mel, min=1e-5))
        return logmel


melspectrogram = LogMelSpectrogram()


def process(i_wave_path, o_mel_path_stem):
    """Convert the waveform into a log-mel-spectrogram and save it in the path.

    For SoftVC-AM training, no needs of the waveform.
    """

    # Load
    wav, sr = torchaudio.load(i_wave_path)
    wav = resample(wav, sr, 16000).unsqueeze(0)

    # wave2logmel
    logmel = melspectrogram(wav).squeeze().numpy()

    # Save
    np.save(o_mel_path_stem.with_suffix(".npy"), logmel)


def run_preprocessing(args):
    """Convert all waveforms under the directory into log-mel-spectrograms."""

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # wave-to-logmel
    print(f"Extracting features for {args.in_dir}")
    # All **/*.wav under the directory
    for i_wave_path in args.in_dir.rglob("*.wav"):
        relative_path = i_wave_path.relative_to(args.in_dir)
        o_mel_path_stem = args.out_dir / relative_path.with_suffix("")

        o_mel_path_stem.parent.mkdir(parents=True, exist_ok=True)
        process(i_wave_path, o_mel_path_stem)
    # /

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess an audio dataset.")
    parser.add_argument("in_dir",  metavar="in-dir",  help="path to the directory containing input waveforms", type=Path)
    parser.add_argument("out_dir", metavar="out-dir", help="path to the output directory",                     type=Path)
    args = parser.parse_args()

    run_preprocessing(args)
