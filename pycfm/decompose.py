from __future__ import division
import transform
import model
import numpy as np


def process(
    signal,
    rate,
    n=1024,
    thop=512,
    alpha=1,
    J=2,
    W=(10, 10),
    mhop=(5, 5),
    verbose=False,
):
    '''Applies CFM Separation

    Args:
       signal (ndarray):    Single or multichannel audio signal
       rate (float):        sample rate
       pref (object):       Preference object


    '''
    print 'computing STFT'
    xstft = transform.cft(signal, n, thop)

    # compute modulation STFT
    print 'computing modulation STFT'
    x = transform.cft(xstft, W, mhop, real=False)

    print 'getting modulation spectrogram, shape:', x.shape
    z = np.abs(x) ** alpha

    cfm = model.CFM(z, nb_iter=10, nb_components=J).fit()

    (P, At, Ac) = cfm.factors

    z_hat = cfm.approx

    # source estimates
    estimates = []
    for j in range(J):

        print 'component %d/%d: separation' % (j+1, J)
        Fj = model.hat(
            P[..., j][..., None],
            At[..., j][..., None],
            Ac[..., j][..., None],
        )

        yj = Fj / z_hat * x

        print 'component %d/%d: reconstructing waveform' % (j+1, J)

        # first compute back STFT
        yjstft = transform.icft(
            yj, fdim=2, hop=mhop, shape=xstft.shape, real=False
        )
        # then waveform
        wavej = transform.icft(
            yjstft, fdim=1, hop=thop, shape=signal.shape
        )
        estimates.append(wavej)

    estimates = np.array(estimates)
    estimates = estimates[:, :signal.shape[0], ...]

    return estimates
