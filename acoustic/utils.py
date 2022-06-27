import torch
import torch.nn.functional as F
import matplotlib

import torchaudio.transforms as transforms

matplotlib.use("Agg")
import matplotlib.pylab as plt


class Metric:
    def __init__(self):
        self.steps = 0
        self.value = 0

    def update(self, value):
        self.steps += 1
        self.value += (value - self.value) / self.steps
        return self.value

    def reset(self):
        self.steps = 0
        self.value = 0


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


def save_checkpoint(
    checkpoint_dir,
    acoustic,
    optimizer,
    step,
    loss,
    best,
    logger,
):
    """Save model/optim/step/loss in the directory.

    Best model is saved also as `model-best.pt`.
    """
    state = {
        #                 ddp.module
        "acoustic-model": acoustic.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "loss": loss,
    }
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / f"model-{step}.pt"
    torch.save(state, checkpoint_path)
    if best:
        best_path = checkpoint_dir / "model-best.pt"
        torch.save(state, best_path)
    logger.info(f"Saved checkpoint: {checkpoint_path.stem}")


def load_checkpoint(
    load_path,
    acoustic,
    optimizer,
    rank,
    logger,
):
    """Restore model/optim/step/loss from the checkpoint."""
    logger.info(f"Loading checkpoint from {load_path}")
    checkpoint = torch.load(load_path, map_location={"cuda:0": f"cuda:{rank}"})
    acoustic.module.load_state_dict(checkpoint["acoustic-model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["step"], checkpoint["loss"]


def plot_spectrogram(spectrogram):
    """Plot spectrogram in 2D figure."""
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig
