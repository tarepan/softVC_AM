import torch
import matplotlib

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
