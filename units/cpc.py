# From https://github.com/facebookresearch/fairseq/tree/main/examples/textless_nlp/gslm/speech2unit
# under MIT License

import soundfile as sf
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray


class CpcFeatureReader:
    """
    Wrapper class to run inference on CPC model.
    Helps extract features for a given audio file.
    """

    def __init__(
        self,
        checkpoint_path,
        layer,
        use_encoder_layer=False,
        norm_features=False,
        sample_rate=16000,
        max_chunk: int = 64000,
    ):
        """
        Args:
            max_chunk - Waveform chunk size
        """
        self.model = load_cpc_model(checkpoint_path, layer).eval().cuda()
        self.sample_rate = sample_rate
        self.max_chunk = max_chunk
        self.norm_features = norm_features
        self.use_encoder_layer = use_encoder_layer

    def read_audio(self, path, ref_len=None) -> NDArray:
        """Read audio from the path.

        Args:
            path - Path to the audio file
            ref_len - ?
        Returns:
            wav :: (T_wave,) - The waveform
        """
        # Load
        wav, sr = sf.read(path)

        # stereo to mono
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim

        # No auto-resampling
        assert sr == self.sample_rate, sr

        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            print(f"ref {ref_len} != read {len(wav)} ({path})")

        return wav

    def get_feats(self, file_path, ref_len=None) -> Tensor:
        """
        Wave-to-Unit conversion from the file.

        Args:
            file_path - Path to the audio file
        Returns:
            :: (T_feat_full, Feat) - Unit series
        """
        # (T_wave,)
        x = self.read_audio(file_path, ref_len)

        # Inspired from CPC_audio feature_loader.py
        with torch.no_grad():
            x = torch.from_numpy(x).float().cuda()
            # (T_wave,) -> (B=1, 1, T_wave)
            x = x.view(1, 1, -1)
            size = x.size(2)

            # feat :: Tensor(B, T_feat, Feat)[]
            feat = []
            start = 0
            while start < size:
                # Chunk-wise feature-nization

                if start + self.max_chunk > size:
                    # End of wave
                    break

                # len(x_chunk) == self.max_chunk
                x_chunk = x[..., start : start + self.max_chunk]

                # chunk_wave::(B, 1, T_wave) to chunk_feat_series::(B, T_feat, Feat)
                feat_chunk = self.model.extract_features(
                    source=x_chunk,
                    get_encoded=self.use_encoder_layer,
                    norm_output=self.norm_features,
                )
                feat.append(feat_chunk)

                # Go to next chunk
                start += self.max_chunk

            if start < size:
                # Tail
                x_chunk = x[:, -self.max_chunk :]
                feat_chunk = self.model.extract_features(
                    source=x_chunk,
                    get_encoded=self.use_encoder_layer,
                    norm_output=self.norm_features,
                )
                df = x_chunk.size(2) // feat_chunk.size(1)
                delta = (size - start) // df
                feat.append(feat_chunk[:, -delta:])

        # Tensor(B=1, T_feat_chunk, Feat)[] -> (B=1, T_feat_full, Feat) -> (T_feat_full, Feat)
        return torch.cat(feat, 1).squeeze(0)


def load_cpc_model(checkpoint_path, layer=None):
    """
    Args:
        layer - The number of LSTM layer
    """
    state_dict = torch.load(checkpoint_path)
    weights = state_dict["weights"]
    config = state_dict["config"]
    if layer is not None:
        config["nLevelsGRU"] = layer

    encoder = CPCEncoder(config["hiddenEncoder"])
    ar_net = CPCAR(
        config["hiddenEncoder"], config["hiddenGar"], False, config["nLevelsGRU"]
    )

    model = CPCModel(encoder, ar_net)
    model.load_state_dict(weights, strict=False)
    model.config = config

    return model


class ChannelNorm(nn.Module):
    def __init__(self, num_features, epsilon=1e-05, affine=True):
        super(ChannelNorm, self).__init__()
        if affine:
            self.weight = nn.parameter.Parameter(torch.Tensor(1, num_features, 1))
            self.bias = nn.parameter.Parameter(torch.Tensor(1, num_features, 1))
        else:
            self.weight = None
            self.bias = None
        self.epsilon = epsilon
        self.p = 0
        self.affine = affine
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        cum_mean = x.mean(dim=1, keepdim=True)
        cum_var = x.var(dim=1, keepdim=True)
        x = (x - cum_mean) * torch.rsqrt(cum_var + self.epsilon)
        if self.weight is not None:
            x = x * self.weight + self.bias
        return x


class CPCEncoder(nn.Module):
    """Convolutional Wave-to-FeatSeries Encoder."""
    def __init__(self, hidden_dim: int = 512):
        """
        Args:
            hidden_dim - Size of Conv_Channel / Output_Feature dimension
        """
        super().__init__()
        # Use `stride`, so cannot use `padding='same'`, but set as padding=same
        self.conv0 = nn.Conv1d(1,          hidden_dim, 10, stride=5, padding=3)
        self.batchNorm0 = ChannelNorm(hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim,  8, stride=4, padding=2)
        self.batchNorm1 = ChannelNorm(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim,  4, stride=2, padding=1)
        self.batchNorm2 = ChannelNorm(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim,  4, stride=2, padding=1)
        self.batchNorm3 = ChannelNorm(hidden_dim)
        self.conv4 = nn.Conv1d(hidden_dim, hidden_dim,  4, stride=2, padding=1)
        self.batchNorm4 = ChannelNorm(hidden_dim)
        self.DOWNSAMPLING = 160

    def get_output_dim(self):
        """Get size of feature dimension."""
        return self.conv4.out_channels

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x :: (B,    1, T_wave) - Waveform
        Returns:
            x :: (B, Feat, T_feat) - Feature series
        """
        x = F.relu(self.batchNorm0(self.conv0(x)))
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = F.relu(self.batchNorm3(self.conv3(x)))
        x = F.relu(self.batchNorm4(self.conv4(x)))
        return x


class CPCAR(nn.Module):
    def __init__(self, dim_encoded: int, dim_output: int, keep_hidden: bool, num_layers: int):
        """
        Args:
            dim_encoded - Size of input feature dimension
            dim_output - Size of hidden/output dimension
            keep_hidden - Whether to keep RNN state over inference
            num_layers - The number of RNN layers
        """
        super().__init__()
        self.baseNet = nn.LSTM(dim_encoded, dim_output, num_layers=num_layers, batch_first=True)
        # RNN state kept over inference
        self.hidden = None
        # Whether to keep RNN state
        self.keep_hidden = keep_hidden

    def get_output_dim(self):
        """Get size of feature dimension."""
        return self.baseNet.hidden_size

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x :: (B, T_feat, Feat_i) - Feature_i series
        Returns:
            x :: (B, T_feat, Feat_o) - Feature_o series
        """
        try:
            self.baseNet.flatten_parameters()
        except RuntimeError:
            pass

        # The forward
        x, h = self.baseNet(x, self.hidden)

        # State preservation over state
        if self.keep_hidden:
            if isinstance(h, tuple):
                # for LSTM?
                self.hidden = tuple(x.detach() for x in h)
            else:
                # for RNN and GRU?
                self.hidden = h.detach()

        return x


class CPCModel(nn.Module):
    def __init__(self, encoder: CPCEncoder, ar_net: CPCAR):
        super().__init__()
        self.gEncoder = encoder
        self.gAR = ar_net
        self.config = None

    def forward(self, x: Tensor, label=None):
        """
        Args:
            x :: (B, 1, T_wave) - The waveform
            label - Pass through
        Returns:
            cpc_feature :: (B, T_feat, Feat_lstm) - Unit series yielded from Conv-LSTM
            encoded     :: (B, T_feat, Feat_conv) - Unit series yielded from Conv
            label                                 - Pass through
        """
        # (B, 1, T_wave) -> (B, Feat, T_feat) -> (B, T_feat, Feat)
        encoded = self.gEncoder(x).permute(0, 2, 1)
        # (B, T_feat, Feat_i) -> (B, T_feat, Feat_o)
        cpc_feature = self.gAR(encoded)
        return cpc_feature, encoded, label

    def extract_features(self, source: Tensor, get_encoded: bool = False, norm_output=False) -> Tensor:
        """
        Args:
            source :: (B, 1, T_wave) - The source waveform
            get_encoded - Whether acquire Conv output or LSTM output (default: LSTM output)
            norm_output - Whether to normalize output over time
        Returns:
            feat_series :: (B, T_feat, Feat) - Continuous unit series
        """
        o_lstm, o_conv, _ = self.forward(source)

        # Feature selection :: (B, T_feat, Feat=feat_conv|feat_lstm)
        feat_series = o_lstm
        if get_encoded:
            feat_series = o_conv

        # Normalize over time
        if norm_output:
            mean = feat_series.mean(dim=1, keepdim=True)
            var = feat_series.var(dim=1, keepdim=True)
            feat_series = (feat_series - mean) / torch.sqrt(var + 1e-08)

        return feat_series
