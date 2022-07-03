from pathlib import Path

from tqdm import tqdm
import torch
from torch import device, from_numpy
import numpy as np
from numpy.typing import NDArray
import s3prl.hub as hub
import librosa


def encode_vqwav2vec_dataset(model: str, in_dir: Path, out_dir: Path, extension: str = ".wav") -> None:
    """
    Wave-to-Unit encoding toward all waveform files under the directory.

    Args:
        model - soft/discrete flag
        in_dir - Directory under which input waveforms exist
        out_dir - Directory under which generated unit series saved
        extension - file extension
    Outputs:
        units :: (T_unit, EmbGroup) - unit series as .npy file
    """

    assert model == "discrete", "This function support only vq-wav2vec discrete."

    _device = device("cuda") if torch.cuda.is_available() else device("cpu")
    encoder = getattr(hub, 'vq_wav2vec')().to(_device)

    print(f"Encoding dataset at {in_dir}")
    # All **/*.{extension} under the directory
    for in_path in tqdm(list(in_dir.rglob(f"*{extension}"))):
        in_path: Path = in_path
        wave: NDArray[np.float32] = librosa.load(str(in_path), sr=16000)[0] # type: ignore ; because of librosa
        with torch.inference_mode():
            # vq-wav2vec do not pad. Manual padding in both side is needed.
            n_receptive_field = 465
            pad_oneside = (n_receptive_field -1) // 2
            wave = np.pad(wave, pad_oneside, mode="reflect") # pyright: ignore [reportUnknownMemberType] ; because of numpy
            i_wave = [from_numpy(wave).to(_device)]
            # [(pad+T_wave+pad,)] => (B=1, T_unit, EmbGroup=2) => (T_unit, EmbGroup)
            unit_series = encoder(i_wave)["codeids"][0]

        out_path = out_dir / in_path.relative_to(in_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path.with_suffix(".npy"), unit_series.cpu().numpy())
