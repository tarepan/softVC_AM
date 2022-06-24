import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class MelDataset(Dataset):
    def __init__(self, root, train=True, discrete=False):
        """
        Prerequisites:
            - log-mel-spectrogram .npy under f'{root}/mels' directory
            - unit series .npy under f'{root}/{'discrete'|'soft'}' directory
        """
        self.discrete = discrete
        self.mels_dir = root / "mels"
        self.units_dir = root / "discrete" if discrete else root / "soft"

        split = "train.txt" if train else "validation.txt"
        with open(root / split) as file:
            self.metadata = [line.strip() for line in file]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        """
        Returns:
            mspc_m1_T :: (T_mspc=2*T_unit+1, Freq) - Mel-spectrogram, from t=-1 (zero padded) to t=T
            unit_0_T  :: (T_unit,            ?)    - Unit series, from t=0 to t=T
        """
        path = self.metadata[index]
        mel_path = self.mels_dir / path
        units_path = self.units_dir / path

        mspc_series = np.load(mel_path.with_suffix(".npy")).T
        unit_0_T = np.load(units_path.with_suffix(".npy"))

        length = 2 * unit_0_T.shape[0]

        # log-mel-spectrogram :: (T_mspc = 2*T_unit+1, Freq)
        mspc_0_T = torch.from_numpy(mspc_series[:length, :])
        ## Zero padding of time for AR input :: (T_mspc = 2*T_unit, Freq) -> (T_mspc = 2*T_unit+1, Freq)
        mspc_m1_T = F.pad(mspc_0_T, (0, 0, 1, 0))

        # unit series :: (T_unit, Feat) ?
        unit_0_T = torch.from_numpy(unit_0_T)
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
            units, batch_first=True, padding_value=100 if self.discrete else 0
        )

        return mels, mels_lengths, units, units_lengths
