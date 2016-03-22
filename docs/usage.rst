Usage
=====

Applying the Common Fate Transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

  import commonfate

  # # forward transform

  # STFT Parameters

  framelength = 1024
  hopsize = 256
  X = commonfate.transform.forward(signal, framelength, hopsize)

  # Patch Parameters
  W = (32, 48)
  mhop = (16, 24)

  Z = commonfate.transform.forward(X, W, mhop, real=False)

  # inverse transform of cft
  Y = commonfate.transform.inverse(
      Z, fdim=2, hop=mhop, shape=X.shape, real=False
  )
  # back to time domain
  y = commonfate.transform.inverse(
      Y, fdim=1, hop=hopsize, shape=x.shape
  )

Fitting the Common Fate Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

  import commonfate

  # initialiase and fit the common fate model
  cfm = commonfate.model.CFM(z, nb_components=10, nb_iter=100).fit()

  # get the fitted factors
  (A, H, C) = cfm.factors

  # returns the of z approximation using the fitted factors
  z_hat = cfm.approx()

Decompose an audio signal using CFT and CFM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*commonfate* has a built-in wrapper which computes the *Common Fate
Transform*, fits the model according to the *Common Fate Model* and
return the synthesised time domain signal components obtained through
wiener / soft mask filtering.

The following example requires to install
`pysoundfile <https://github.com/bastibe/PySoundFile>`__.

.. code:: python

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
      "comp_3.wav",
      components[2, ...],
      fs
  )
