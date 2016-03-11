# _Common Fate Model_ for Audio Source Separation

[![Build Status](https://travis-ci.org/faroit/pyCFM.svg?branch=master)](https://travis-ci.org/faroit/pyCFM)

## Optimisations

The current common fate model implementation makes heavily use of the [Einstein Notation](https://en.wikipedia.org/wiki/Einstein_notation). We use the [numpy ```einsum``` module](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.einsum.html
) which can be slow on large tensors. To speed up the computation time we recommend to install [Daniel Smith's ```opt_einsum``` package](https://github.com/dgasmith/opt_einsum).

##### Installation via pip
```bash
pip install -e 'git+https://github.com/dgasmith/opt_einsum.git#egg=opt_einsum'
```

_PyCFM_ automatically detects if the package is installed.

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
