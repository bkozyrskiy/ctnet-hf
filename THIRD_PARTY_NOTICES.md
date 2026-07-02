# Third-party notices and data terms

CTNet-HF is an original clean-room Hugging Face implementation of the CTNet
architecture described in:

> Zhao et al., “CTNet: a convolutional transformer network for EEG-based motor
> imagery classification,” *Scientific Reports* 14, 20237 (2024).
> https://doi.org/10.1038/s41598-024-71118-7

The authors' reference implementation is available at
https://github.com/snailpt/CTNet. CTNet-HF does not copy or redistribute that
repository's source code or training checkpoints.

The Python dependencies used by CTNet-HF, including PyTorch, Transformers,
NumPy, scikit-learn, MOABB, and MNE, remain under their respective licenses.
They are dependencies and are not vendored in this repository or its model
bundles.

BNCI2014-001 recordings are not redistributed by CTNet-HF. Users who download
the dataset through MOABB or another source are responsible for complying with
the dataset's access, citation, consent, and usage terms. The MIT License for
CTNet-HF source code and released model weights does not sublicense the
underlying EEG recordings.
