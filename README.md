# _Common Fate Model_ for Audio Source Separation

[![Build Status](https://travis-ci.org/aliutkus/cfm.svg?branch=master)](https://travis-ci.org/aliutkus/cfm)

This package is a python implementation of the _common fate transform and model_ as described in [this paper](https://hal.archives-ouvertes.fr/hal-01248012/file/common_fate_icassp2016.pdf).

### Common Fate Transform

The Common Fate Transform is based on a signal representation that divides a complex spectrogram into a grid of patches of arbitrary size. These complex patches are then processed by a two-dimensional discrete Fourier transform, forming a tensor representation which reveals spectral and temporal modulation textures.

### Common Fate Model

An adapted factorization model similar to the PARAFAC/CANDECOMP factorisation allows to decompose the _common fate transform_ tesnor into different time-varying harmonic sources based on their particular common modulation profile: hence the name Common Fate Model.

![cfm](https://cloud.githubusercontent.com/assets/72940/13718782/b1a331e4-e7ed-11e5-9612-0cacd6fe34a9.png)

## Usage

```python
import commonfate

# forward transform
X = commonfate.transform.forward(signal, framelength, hopsize)
Z = commonfate.transform.forward(X, W, mhop, real=False)

# inverse transform of cft
Y = commonfate.transform.inverse(
    Z, fdim=2, hop=mhop, shape=X.shape, real=False
)
# back to time domain
y = commonfate.transform.inverse(
    Y, fdim=1, hop=hopsize, shape=x.shape
)

```

## Optimisations

The current common fate model implementation makes heavily use of the [Einstein Notation](https://en.wikipedia.org/wiki/Einstein_notation). We use the [numpy ```einsum``` module](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.einsum.html
) which can be slow on large tensors. To speed up the computation time we recommend to install [Daniel Smith's ```opt_einsum``` package](https://github.com/dgasmith/opt_einsum).

##### Installation via pip
```bash
pip install -e 'git+https://github.com/dgasmith/opt_einsum.git#egg=opt_einsum'
```

_commonfate_ automatically detects if the package is installed.

## References

You can download and read the paper [here](https://hal.archives-ouvertes.fr/hal-01248012/file/common_fate_icassp2016.pdf).
If you use this package, please reference to the following publication:

```tex
@inproceedings{stoeter2016cfm,
  TITLE = {{Common Fate Model for Unison source Separation}},
  AUTHOR = {St{\"o}ter, Fabian-Robert and Liutkus, Antoine and Badeau, Roland and Edler, Bernd and Magron, Paul},
  BOOKTITLE = {{41st International Conference on Acoustics, Speech and Signal Processing (ICASSP)}},
  ADDRESS = {Shanghai, China},
  PUBLISHER = {{IEEE}},
  SERIES = {Proceedings of the 41st International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  YEAR = {2016},
  KEYWORDS = {Non-Negative tensor factorization ; Sound source separation ; Common Fate Model},
}
```
