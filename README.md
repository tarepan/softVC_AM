<div align="center">

# SoftVC-AM : Unit-to-Mspc VC module of SoftVC <!-- omit in toc -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][notebook]
[![Paper](http://img.shields.io/badge/paper-arxiv.2110.06280-B31B1B.svg)][paper]

</div>

AcousticModel module of SoftVC, unit-based Any-to-One voice conversion.  
This repository is a fork of official [SoftVC-AM][official_softvcam].  

For more whole-model details, samples and E2E demo, see [soft-vc](https://github.com/bshall/soft-vc).

<div align="center">
    <img width="100%" alt="Soft-VC"
      src="https://raw.githubusercontent.com/bshall/acoustic-model/main/acoustic-model.png">
</div>
<div>
  <sup>
    <strong>Fig 1:</strong> Architecture of the voice conversion system. a) The <strong>discrete</strong> content encoder clusters audio features to produce a sequence of discrete speech units. b) The <strong>soft</strong> content encoder is trained to predict the discrete units. The acoustic model transforms the discrete/soft speech units into a target spectrogram. The vocoder converts the spectrogram into an audio waveform.
  </sup>
</div>

## How to Use
### Inference
#### As Library
```python
import torch
import numpy as np

# Load checkpoint (either hubert_soft or hubert_discrete)
acoustic = torch.hub.load("bshall/acoustic-model:main", "hubert_soft").cuda()

# Load speech units
units = torch.from_numpy(np.load("path/to/units"))

# Generate mel-spectrogram
mel = acoustic.generate(units)
```

#### As Script

```
usage: generate.py [-h] {soft,discrete} in-dir out-dir

Generate spectrograms from input speech units (discrete or soft).

positional arguments:
  {soft,discrete}  available models (HuBERT-Soft or HuBERT-Discrete)
  in-dir           path to the dataset directory.
  out-dir          path to the output directory.

optional arguments:
  -h, --help       show this help message and exit
```

### Training
☞ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][notebook]  

#### Preprocessing
wave-to-mel preprocessing:  
```bash
python mels.py <in-dir> <out-dir>
```

#### Train
unit-to-mel training:
```bash
python train.py <dataset-dir> <checkpoint-dir>
# python train.py <dataset-dir> <checkpoint-dir> --resume=<checkpoint-file> # Resume
```

## Links

- [Soft-VC root repo](https://github.com/bshall/soft-vc)
  - [Soft-VC wave-to-unit HuBERT](https://github.com/bshall/hubert)
  - [Soft-VC mspc-to-wave HiFiGAN](https://github.com/bshall/hifigan)
- [Soft-VC paper](https://ieeexplore.ieee.org/abstract/document/9746484)

## References
### Original paper
```
@inproceedings{
    soft-vc-2022,
    author={van Niekerk, Benjamin and Carbonneau, Marc-André and Zaïdi, Julian and Baas, Matthew and Seuté, Hugo and Kamper, Herman},
    booktitle={ICASSP}, 
    title={A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion}, 
    year={2022}
}
```

[paper]: https://arxiv.org/abs/2111.02392
[notebook]: https://colab.research.google.com/github/tarepan/softVC_AM/blob/main/softVC_AM.ipynb
[official_softvcam]: https://github.com/bshall/acoustic-model