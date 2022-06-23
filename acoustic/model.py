import torch
import torch.nn as nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

URLS = {
    "hubert-discrete": "https://github.com/bshall/acoustic-model/releases/download/v0.1/hubert-discrete-ffc42c75.pt",
    "hubert-soft": "https://github.com/bshall/acoustic-model/releases/download/v0.1/hubert-soft-aa8d82f5.pt",
}


class AcousticModel(nn.Module):
    """Unit-to-Mel acoustic model, SegFC-Conv-AR(SegFC-Res(LSTM)-Proj)"""
    def __init__(self, discrete: bool = False, upsample: bool = True):
        """
        Args:
            discrete - Whether input is discrete unit or not (affect only Encoder embedding ON/OFF)
            upsample -
        """
        super().__init__()
        # [Emb-]SegFC-Conv
        self.encoder = Encoder(discrete, upsample)
        # AR(SegFC-Res(LSTM)-Proj)
        self.decoder = Decoder()

    def forward(self, unit_series: torch.Tensor, mels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            unit_series - Unit series
            mels - Ground Truth mel-spectrograms for teacher-forcing
        Returns - Mel-spectrograms
        """
        latent_series = self.encoder(unit_series)
        return self.decoder(latent_series, mels)

    @torch.inference_mode()
    def generate(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.decoder.generate(x)


class Encoder(nn.Module):
    """Unit-to-Latent FeedForward Encoder, [Emb-]SegFC-Conv"""
    def __init__(self, discrete: bool = False, upsample: bool = True):
        super().__init__()
        dim_z = 512

        self.embedding = nn.Embedding(100 + 1, 256) if discrete else None
        # (SegFC256-ReLU-DO0.5)x2
        self.prenet = PreNet(256, 256, 256)
        # (Conv-ReLU-IN)-ConvT-(Conv-ReLU-IN)x2
        self.convs = nn.Sequential(
            nn.Conv1d(256, dim_z, 5, 1, 2),
            nn.ReLU(),
            nn.InstanceNorm1d(dim_z),
            nn.ConvTranspose1d(dim_z, dim_z, 4, 2, 1) if upsample else nn.Identity(),
            nn.Conv1d(dim_z, dim_z, 5, 1, 2),
            nn.ReLU(),
            nn.InstanceNorm1d(dim_z),
            nn.Conv1d(dim_z, dim_z, 5, 1, 2),
            nn.ReLU(),
            nn.InstanceNorm1d(dim_z),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.embedding is not None:
            x = self.embedding(x)
        x = self.prenet(x)
        x = self.convs(x.transpose(1, 2))
        return x.transpose(1, 2)


class Decoder(nn.Module):
    """Latent-to-Mel autoregressive Decoder, AR(SegFC-Res(LSTM)-Proj)"""
    def __init__(self):
        super().__init__()
        n_mels = 128
        dim_z = 512
        dim_ar = 256
        dim_h_lstm = 768

        self.prenet = PreNet(n_mels, 256, dim_ar) # (SegFC256-ReLU-DO0.5)x2
        self.lstm1 =  nn.LSTM(dim_z + dim_ar, dim_h_lstm, batch_first=True)
        self.lstm2 =  nn.LSTM(dim_h_lstm,     dim_h_lstm, batch_first=True)
        self.lstm3 =  nn.LSTM(dim_h_lstm,     dim_h_lstm, batch_first=True)
        self.proj = nn.Linear(dim_h_lstm,     n_mels, bias=False)

    def forward(self, x: torch.Tensor, mels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -
            mels - Ground Truth mel-spectrograms for teacher-forcing
        Returns - Estimated mel-spectrograms
        """
        mels = self.prenet(mels)
        x, _ = self.lstm1(torch.cat((x, mels), dim=-1))
        res = x
        x, _ = self.lstm2(x)
        x = res + x
        res = x
        x, _ = self.lstm3(x)
        x = res + x
        return self.proj(x)

    @torch.inference_mode()
    def generate(self, xs: torch.Tensor) -> torch.Tensor:
        m = torch.zeros(xs.size(0), 128, device=xs.device)
        h1 = torch.zeros(1, xs.size(0), 768, device=xs.device)
        c1 = torch.zeros(1, xs.size(0), 768, device=xs.device)
        h2 = torch.zeros(1, xs.size(0), 768, device=xs.device)
        c2 = torch.zeros(1, xs.size(0), 768, device=xs.device)
        h3 = torch.zeros(1, xs.size(0), 768, device=xs.device)
        c3 = torch.zeros(1, xs.size(0), 768, device=xs.device)

        mel = []
        for x in torch.unbind(xs, dim=1):
            m = self.prenet(m)
            x = torch.cat((x, m), dim=1).unsqueeze(1)
            x1, (h1, c1) = self.lstm1(x, (h1, c1))
            x2, (h2, c2) = self.lstm2(x1, (h2, c2))
            x = x1 + x2
            x3, (h3, c3) = self.lstm3(x, (h3, c3))
            x = x + x3
            m = self.proj(x).squeeze(1)
            mel.append(m)
        return torch.stack(mel, dim=1)


class PreNet(nn.Module):
    """Encoder/Decoder PreNet, (SegFC-ReLU-DO)x2."""
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _acoustic(
    name: str,
    discrete: bool,
    upsample: bool,
    pretrained: bool = True,
    progress: bool = True,
) -> AcousticModel:
    """Generate AcousticModel instance.

    Args:
        name :: "hubert-discrete" | "hubert-soft" - Model name
        upsample - (Currently always ON)
    """
    acoustic = AcousticModel(discrete, upsample)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(URLS[name], progress=progress)
        consume_prefix_in_state_dict_if_present(checkpoint, "module.")
        acoustic.load_state_dict(checkpoint)
        acoustic.eval()
    return acoustic


def hubert_discrete(
    pretrained: bool = True,
    progress: bool = True,
) -> AcousticModel:
    r"""HuBERT-Discrete acoustic model from `"A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion"`.
    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
    """
    return _acoustic(
        "hubert-discrete",
        discrete=True,
        upsample=True,
        pretrained=pretrained,
        progress=progress,
    )


def hubert_soft(
    pretrained: bool = True,
    progress: bool = True,
) -> AcousticModel:
    r"""HuBERT-Soft acoustic model from `"A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion"`.
    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
    """
    return _acoustic(
        "hubert-soft",
        discrete=False,
        upsample=True,
        pretrained=pretrained,
        progress=progress,
    )
