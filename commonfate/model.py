import numpy as np
import tqdm
try:
    from opt_einsum import contract as einsum
except ImportError:
    from numpy import einsum


def hat(A, H, C, eps=None):
    """Builds common fate model tensor from factors
    makes use of numpys einstein summation

    Parameters
    ----------
    A : tuple
        The shape
    H : tuple
        The shape
    C : tuple
        The shape

    Returns
    -------
    np.ndarray, shape=(P.shape + H.shape + C.shape)
        tensor with same shape as common fate transform
    """
    if eps is None:
        eps = np.finfo(float).eps

    return eps + einsum('abfj,tj,cj->abftc', A, H, C)


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
    Pj(a,b,f,t,c) = A(a,b,f,j)H(t,j)C(c,j)

    Factorises one modulation texture P for each frequency,
    hence A(a,b,f,j) which is activated over time, this is H(t,j) and over
    channels C(c,j)

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

    H : ndarray, optional
        initialisation of H. Defaults to `none`,
        results in random initialisation. shape=(t, j)

    C : ndarray, optional
        initialisation of C. Defaults to `none`,
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
        A=None,
        H=None,
        C=None,
    ):
        # General fitting parameters
        self.data = data
        self.nb_components = nb_components
        self.beta = float(beta)
        self.nb_iter = nb_iter

        # Factorisation Parameters
        if A is None:
            self._A = nnrandn(self.data.shape[:3] + (nb_components,))
        else:
            self._A = A

        if H is None:
            self._H = nnrandn((self.data.shape[3], nb_components))
        else:
            self._H = H

        if C is None:
            self._C = nnrandn((self.data.shape[4], nb_components))
        else:
            self._C = C

    def fit(self):
        """fits a common fate model

        returns ``CFM`` model
        """

        def MU(einsumString, Z, factors):
            Zhat = hat(self._A, self._H, self._C)
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
            self._A *= MU('abftc,tj,cj->abfj', self.data, (self._H, self._C))
            self._H *= MU('abftc,abfj,cj->tj', self.data, (self._A, self._C))
            self._C *= MU('abftc,abfj,tj->cj', self.data, (self._A, self._H))

        return self

    @property
    def factors(self):
        """
        Returns Common Fate factors

        :type: tuple
        """
        return (self._A, self._H, self._C)

    def approx(self):
        """
        Builds Common Fate tensor from fitted model

        returns ndarray, shape=(a, b, f, t, c)
        """
        return hat(self._A, self._H, self._C)
