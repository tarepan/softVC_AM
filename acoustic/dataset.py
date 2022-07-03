from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class MelUnitDataset(Dataset):
    """Dataset yielding unit series and mel-spectrograms."""
    def __init__(self, root: Path, train: bool = True, discrete: bool = False, upsample: bool = True, size_codebook: int = 1):
        """
        Prerequisites:
            - log-mel-spectrogram
            - unit series
        Data structure:
            root/
                mels/
                    <rel_path_in_metadata>.npy
                soft|discrete/
                    <rel_path_in_metadata>.npy
        Args:
            root - Data root path, under which data exist
            train - Whether train mode or not
            discrete - Whether discrete unit or not (default: soft)
            upsample - Whether to upsample unit
            size_codebook - Size of single codebook
        """
        self.discrete = discrete
        self.size_codebook = size_codebook
        self.mels_dir = root / "mels"
        self.units_dir = root / "discrete" if discrete else root / "soft"

        split = "train.txt" if train else "validation.txt"
        with open(root / split) as file:
            self.metadata = [line.strip() for line in file]

        self.upsampling_rate = 2 if upsample else 1

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        """
        Returns:
            mspc_m1_T :: (T_mspc=2*T_unit+1, Freq)  - Mel-spectrogram, from t=-1 (zero padded) to t=T
            unit_0_T  :: (T_unit, Feat) | (T_unit,) - Continuous | Discrete Unit series, from t=0 to t=T
        """
        path = self.metadata[index]
        path_mspc_series = self.mels_dir  / path
        path_unit_series = self.units_dir / path

        mspc_series: NDArray = np.load(path_mspc_series.with_suffix(".npy")).T
        unit_series: NDArray = np.load(path_unit_series.with_suffix(".npy"))

        # Length adjustment (adjust to short one)
        len_unit_melscale = self.upsampling_rate * unit_series.shape[0]
        len_mspc          =                        mspc_series.shape[0]
        if len_unit_melscale <= len_mspc:
            # Unit <= Mspc, so adjust mspc to unit
            mspc_0_T = mspc_series[:len_unit_melscale]
            unit_0_T = unit_series
        else:
            # Mspc < Unit, so adjust to mspc
            # todo: Non-integer upsampling rate
            len_mspc_just = len_mspc // self.upsampling_rate * self.upsampling_rate
            len_unit = len_mspc_just // self.upsampling_rate
            mspc_0_T = mspc_series[:len_mspc_just]
            unit_0_T = unit_series[:len_unit]
            
        mspc_0_T = torch.from_numpy(mspc_0_T)
        unit_0_T = torch.from_numpy(unit_0_T)

        ## Zero padding of time for AR input :: (T_mspc = 2*T_unit, Freq) -> (T_mspc = 2*T_unit+1, Freq)
        mspc_m1_T = F.pad(mspc_0_T, (0, 0, 1, 0))

        if self.discrete:
            unit_0_T = unit_0_T.long()

        return mspc_m1_T, unit_0_T

    def pad_collate(self, batch):
        """collate_fn used in the dataloader.

        Make padding for batching.
        Returns:
            mels          :: (B, max(T_mspc)=2*max(T_unit), Freq) - t=-1 (zero padded) ~ t=T
            mels_lengths
            units         :: (B, max(T_unit),               Feat) ? - t=0 ~ t=T
            units_lengths
        """

        mels, units = zip(*batch)

        mels, units = list(mels), list(units)

        # mel[0] is used only for AR input, not for output/loss
        mels_lengths = torch.tensor([x.size(0) - 1 for x in mels])
        units_lengths = torch.tensor([x.size(0) for x in units])

        mels = pad_sequence(mels, batch_first=True)
        units = pad_sequence(
            units, batch_first=True, padding_value=self.size_codebook if self.discrete else 0
        )

        return mels, mels_lengths, units, units_lengths
