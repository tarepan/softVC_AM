from pathlib import Path

import joblib
from tqdm import tqdm
import torch
import numpy as np

from .cpc import CpcFeatureReader


def encode_cpc_dataset(model: str, in_dir: Path, out_dir: Path, extension: str = ".flac"):
    """Wave-to-Unit encoding toward all waveform files under the directory.
    Outputs:
        units - unit series as .npy file
    """

    path_enc_state = "./cpc_big_ll6kh_top_ctc.pt"
    path_kmeans_state = "./km.bin"

    assert model == "discrete", "This function support only cpc discrete."

    print(f"Loading cpc checkpoint")
    # Models
    ## k-means seems to be trained with default CpcFeatureReader `use_encoder_layer=False, norm_features=False`.
    encoder = CpcFeatureReader(checkpoint_path=path_enc_state, layer=1)
    with open(path_kmeans_state, "rb") as f:
        kmeans_model = joblib.load(f)
    kmeans_model.verbose = False

    print(f"Encoding dataset at {in_dir}")
    # All **/*.{extension} under the directory
    for in_path in tqdm(list(in_dir.rglob(f"*{extension}"))):
        with torch.inference_mode():
            # Wave-to-ContinuousUnit :: path -> NDArray(T_feat_full, Feat)
            continuous_unit_series = encoder.get_feats(str(in_path)).cpu().numpy()

        # ContinuousUnit-to-DiscreteUnit :: NDArray(T_feat, Feat) -> NDArray(T_feat,)
        discrete_unit_series = kmeans_model.predict(continuous_unit_series)

        out_path = out_dir / in_path.relative_to(in_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path.with_suffix(".npy"), discrete_unit_series)
