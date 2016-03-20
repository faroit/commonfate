from __future__ import division
import transform
import model
import numpy as np


def process(
    signal,
    nb_components,
    n_fft=1024,
    n_hop=512,
    cft_patch=(10, 10),
    cft_hop=(5, 5),
    alpha=1,
    nb_iter=100
):
    """Separates an audio signal into `nb_components` components using
    the Common Fate Transform and Common Fate Model. This is essentially a
    wrapper for `model` and `transform` to be used in the context of source
    separation


    Parameters
    ----------
    signal : ndarray, shape (nb_samples, nb_channels)
        input audio signal of `ndim = 2`. Use np.atleast_2d() for mono audio

    nb_components : int
        Number of latent variable components for use in Common Fate Model

    n_fft : int, optional
        FFT window size, defaults to `1024`

    n_hop : int, optional
        FFT hop size, defaults to 512

    cft_patch : tuple(int), optional
        Common Fate transform patch size of shape (a, b),
        where a merges frequency bins and b merges time frames.
        Defaults to `(10, 10)`

    cft_hop : tuple(int), optional
        Common Fate transform hop size of shape (a_hop, b_hop),
        where `a_hop` is the hop size for `a` and `b_hop` for `b`
        Defaults to `(5, 5)`

    alpha : int
        uses cft to the power of alpha (``np.abs(xcft) ** alpha``)
        defaults to `1`

    nb_iter : int
        number of iterations for Common Fate model fit

    Returns
    -------
    ndarray, shape=(component, nb_samples, nb_channels)
        Trensor of output components

    """
    xstft = transform.forward(
        signal,
        n_fft,
        n_hop,
    )

    xcft = transform.forward(
        xstft,
        cft_patch,
        cft_hop,
        real=False
    )

    cfm = model.CFM(
        np.abs(xcft) ** alpha,
        nb_iter=nb_iter,
        nb_components=nb_components
    ).fit()

    (A, H, C) = cfm.factors

    xcft_hat = cfm.approx()

    # source estimates
    estimates = []
    for j in range(nb_components):

        Fj = model.hat(
            A[..., j][..., None],
            H[..., j][..., None],
            C[..., j][..., None],
        )

        yj = Fj / xcft_hat * xcft

        # first compute back STFT
        yjstft = transform.inverse(
            yj, fdim=2, hop=cft_hop, shape=xstft.shape, real=False
        )
        # then waveform
        wavej = transform.inverse(
            yjstft, fdim=1, hop=n_hop, shape=signal.shape
        )
        estimates.append(wavej)

    return np.array(estimates)
