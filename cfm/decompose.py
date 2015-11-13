import mstft
import pylab as plt
import numpy as np
import itertools
from progressbar import ProgressBar, FormatLabel, Bar


"""-----------------------------------------------------------------------------
Common Fate model:
Vj(a,b,f,t,c) = P(a,b,f,j)At(t,j)Ac(c,j)

So we have one modulation texture "shape" for each frequency, hence P(a,b,f,j)
which is activated over time, this is At(t,j) and over channels, this is
Ac(c,j)

--------------------------------------------------------------------------------
Antoine Liutkus, Inria 2015"""


def hat(P, At, Ac, eps=1e-10):
    return eps+np.einsum('abfj,tj,cj->abftc', P, At, Ac)


def MNMFfit(Z, niter, P, At, Ac, beta=1):
    # fits a common fate model to Z(a,b,f,t,i)=P(a,b,j)Af(f,j)At(t,j)Ac(i,j)
    eps = 1e-10

    def MU(einsumString, Z, factors):
        Zhat = hat(P, At, Ac, eps)
        return (np.einsum(einsumString, Z*(Zhat**(beta-2)), *factors) /
                np.einsum(einsumString, Zhat**(beta-1), *factors))

    widgets = [FormatLabel('NTF Iteration %(value)d '), Bar()]
    progress = ProgressBar(widgets=widgets)
    for it in progress(range(niter)):
        P *= MU('abftc,tj,cj->abfj', Z, (At, Ac))
        At *= MU('abftc,abfj,cj->tj', Z, (P, Ac))
        Ac *= MU('abftc,abfj,tj->cj', Z, (P, At))
    return (P, At, Ac)


def nnrandn(shape):
    """generates randomly a nonnegative ndarray of given shape

    Parameters
    ----------
    shape : tuple
        The shape

    Returns
    -------
    out : array of given shape
        The non-negative random numbers
    """
    return np.abs(np.random.randn(*shape))


def displayMSTFT(Z, name=None):
    # display a modulation spectrogram, of shape (w1,w2,f,t)
    plt.figure(1)
    (nF, nT) = Z.shape[2:4]
    for (f, t) in itertools.product(range(nF), range(nT)):
        plt.subplot(nF, nT, (nF-f-1) * nT+t+1)
        plt.imshow(
            abs(Z[..., f, t]) ** 0.3,
            vmin=0,
            vmax=10,
            cmap='jet',
            aspect='auto'
        )
        plt.xticks([])
        plt.xlabel('')
        plt.yticks([])
        plt.ylabel('')
    if name is None:
        plt.show()
    else:
        plt.savefig(name)


def process(signal, rate, pref, verbose=False, cluster=None, display=None):
    '''Applies CFM Separation

    Args:
       signal (ndarray):    Single or multichannel audio signal
       rate (float):        sample rate
       pref (object):       Preference object

    Returns:
       output (ndarray):    output array separated into pref.k components or
                            clustered into pref.s components using kmeans

    '''
    W = (pref['W_A'], pref['W_B'])
    mhop = (pref['mhop_A'], pref['mhop_B'])

    print 'computing STFT'
    xstft = mstft.stft(signal, pref['nfft'], pref['thop'])

    # compute modulation STFT
    print 'computing modulation STFT'
    x = mstft.stft(xstft, W, mhop, real=False)

    print 'getting modulation spectrogram, shape:', x.shape
    z = np.abs(x) ** pref['alpha']

    if display:
        displayMSTFT(z[..., :10, :30, 0])

    # getting dimensions
    (nF, nT, I) = x.shape[2:]

    # common fate model parameters
    P = nnrandn(W+(nF, pref['J']))
    At = nnrandn((nT, pref['J']))
    Ac = nnrandn((I, pref['J']))

    (P, At, Ac) = MNMFfit(z, pref['i'], P, At, Ac)

    eps = 1e-10

    model = hat(P, At, Ac, eps)

    # source estimates
    estimates = []
    for j in range(pref['J']):

        print 'component %d/%d: separation' % (j+1, pref['J'])
        Fj = hat(
            P[..., j][..., None],
            At[..., j][..., None],
            Ac[..., j][..., None],
            eps=eps/float(pref['J'])
        )

        yj = Fj/model*x

        if display:
            displayMSTFT(Fj[..., :10, :80, 0], name='source%d.pdf' % (j+1))

        print 'component %d/%d: reconstructing waveform' % (j+1, pref['J'])

        # first compute back STFT
        yjstft = mstft.istft(
            yj, fdim=2, hop=mhop, shape=xstft.shape, real=False
        )
        # then waveform
        wavej = mstft.istft(
            yjstft, fdim=1, hop=pref['thop'], shape=signal.shape
        )
        estimates.append(wavej)

    estimates = np.array(estimates)
    estimates = estimates[:, :signal.shape[0], ...]

    return estimates
