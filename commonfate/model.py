import numpy as np
import tqdm
try:
    from opt_einsum import contract as einsum
except ImportError:
    from numpy import einsum


def hat(P, At, Ac, eps=None):
    """Builds common fate model tensor from factors
    makes use of numpys einstein summation

    Parameters
    ----------
    P : tuple
        The shape
    At : tuple
        The shape
    Ac : tuple
        The shape

    Returns
    -------
    np.ndarray, shape=(P.shape + At.shape + Ac.shape)
        tensor with same shape as common fate transform
    """
    if eps is None:
        eps = np.finfo(float).eps

    return eps + einsum('abfj,tj,cj->abftc', P, At, Ac)


def nnrandn(shape):
    """generates randomly a nonnegative ndarray of given shape

    Parameters
    ----------
    shape : tuple
        The shape

    Returns
    -------
    out : ndarray
        The non-negative random numbers
    """
    return np.abs(np.random.randn(*shape))


class CFM(object):
    """The Common Fate model
    Vj(a,b,f,t,c) = P(a,b,f,j)At(t,j)Ac(c,j)

    Factorises one modulation texture P for each frequency,
    hence P(a,b,f,j) which is activated over time, this is At(t,j) and over
    channels Ac(c,j)

    Parameters
    ----------

    data : Data input tensor of shape (a, b, f, t, c)
        (a, b): Patch Dimension
        (f, t): Patch frequency and time index
        (c,): Channel

    nb_components : int > 0
        the number of latent components for the CFM model
        positive integer

    nb_iter : int, opt
        number of iterations

    beta : float
        The beta-divergence to use. An arbitrary float, but not
        that non-small integer values will significantly slow the
        calculation down. Particular cases of interest are:

         * beta=2 : Euclidean distance
         * beta=1 : Kullback Leibler
         * beta=0 : Itakura-Saito

    P : ndarray, optional
        initialisation of P. Defaults to `none`,
        results in random initialisation. shape=(a, b, f, j)

    At : ndarray, optional
        initialisation of At. Defaults to `none`,
        results in random initialisation. shape=(t, j)

    Ac : ndarray, optional
        initialisation of Ac. Defaults to `none`,
        results in random initialisation. shape=(c, j)

    Methods
    -------
    fit()
        Fits the model to the given data using `nb_iter` iterations
        returns ``CFM`` model

    """
    def __init__(
        self,
        data,
        nb_components,
        nb_iter=100,
        beta=1,
        P=None,
        At=None,
        Ac=None,
    ):
        # General fitting parameters
        self.data = data
        self.nb_components = nb_components
        self.beta = float(beta)
        self.nb_iter = nb_iter

        # Factorisation Parameters
        if P is None:
            self._P = nnrandn(self.data.shape[:3] + (nb_components,))
        else:
            self._P = P

        if At is None:
            self._At = nnrandn((self.data.shape[3], nb_components))
        else:
            self._At = At

        if Ac is None:
            self._Ac = nnrandn((self.data.shape[4], nb_components))
        else:
            self._Ac = Ac

    def fit(self):
        """fits a common fate model to
        Z(a,b,f,t,i) = P(a,b,j)Af(f,j)At(t,j)Ac(i,j)

        returns ``CFM`` model
        """

        def MU(einsumString, Z, factors):
            Zhat = hat(self._P, self._At, self._Ac)
            return (
                einsum(
                    einsumString,
                    self.data * (Zhat ** (self.beta - 2)),
                    *factors) /
                einsum(
                    einsumString,
                    Zhat ** (self.beta - 1),
                    *factors
                )
            )

        for it in tqdm.tqdm(range(self.nb_iter)):
            self._P *= MU('abftc,tj,cj->abfj', self.data, (self._At, self._Ac))
            self._At *= MU('abftc,abfj,cj->tj', self.data, (self._P, self._Ac))
            self._Ac *= MU('abftc,abfj,tj->cj', self.data, (self._P, self._At))

        return self

    @property
    def factors(self):
        """
        Returns Common Fate factors

        :type: tuple
        """
        return (self._P, self._At, self._Ac)

    def approx(self):
        """
        Builds Common Fate tensor from fitted model

        returns ndarray, shape=(a, b, f, t, c)
        """
        return hat(self._P, self._At, self._Ac)
