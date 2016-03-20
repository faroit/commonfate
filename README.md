# Common Fate Transform and Model for Python

[![Build Status](https://travis-ci.org/aliutkus/commonfate.svg?branch=master)](https://travis-ci.org/aliutkus/commonfate)

This package is a python implementation of the _Common Fate Transform and Model_ to be used for audio source separation as described in [an ICASSP 2016 paper "Common Fate Model for Unison source Separation"](https://hal.archives-ouvertes.fr/hal-01248012/file/common_fate_icassp2016.pdf).

### Common Fate Transform

![cft](https://cloud.githubusercontent.com/assets/72940/13906318/5de230a0-ef0e-11e5-8447-3a2f1600a22a.png)

The Common Fate Transform is based on a signal representation that divides a complex spectrogram into a grid of patches of arbitrary size. These complex patches are then processed by a two-dimensional discrete Fourier transform, forming a tensor representation which reveals spectral and temporal modulation textures.

### Common Fate Model

![cfm](https://cloud.githubusercontent.com/assets/72940/13906456/402211d6-ef11-11e5-8103-12944f5404f4.png)

An adapted factorization model similar to the PARAFAC/CANDECOMP factorisation allows to decompose the _common fate transform_ tesnor into different time-varying harmonic sources based on their particular common modulation profile: hence the name Common Fate Model.

## Usage

### Applying the Common Fate Transform

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

### Fitting the Common Fate Model

```python
import commonfate

# initialiase and fit the common fate model
cfm = commonfate.model.CFM(z, nb_components=10, nb_iter=100).fit()

# get the fitted factors
(A, H, C) = cfm.factors

# returns the of z approximation using the fitted factors
z_hat = cfm.approx()
```

### Decompose an audio signal using CFT and CFM

_commonfate_ has a built-in wrapper which computes the _Common Fate Transform_,
fits the model according to the _Common Fate Model_ and return the synthesised
time domain signal components obtained through wiener / soft mask filtering.

The following example requires to install [pysoundfile](https://github.com/bastibe/PySoundFile).

```python
import commonfate
import soundfile as sf

# loading signal
(audio, fs) = sf.read(filename, always_2d=True)

# decomposes the audio signal into
# (nb_components, nb_samples, nb_channels)
components = decompose.process(
    audio,
    nb_iter=100,
    nb_components=10,
    n_fft=1024,
    n_hop=256,
    cft_patch=(32, 48),
    cft_hop=(16, 24)
)

# write out the third component to wave file
sf.write(
    "component_3.wav",
    components[2, ...],
    fs
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
