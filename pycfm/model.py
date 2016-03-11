import numpy as np
import tqdm
try:
    from opt_einsum import contract as einsum
except ImportError:
    from numpy import einsum


def hat(P, At, Ac, eps=None):
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
    out : array of given shape
        The non-negative random numbers
    """
    return np.abs(np.random.randn(*shape))


class CFM(object):
    """The Common Fate model
    Vj(a,b,f,t,c) = P(a,b,f,j)At(t,j)Ac(c,j)

    So we have one modulation texture "shape" for each frequency,
    hence P(a,b,f,j) which is activated over time, this is At(t,j) and over
    channels, this is Ac(c,j)

    Parameters
    ---------
    data_shape : iterable
        A tuple of integers representing the shape of the
        data to approximate
    n_components : int > 0
        the number of latent components for the NTF model
        positive integer
    beta : float
        The beta-divergence to use. An arbitrary float, but not
        that non-small integer values will significantly slow the
        calculation down. Particular cases of interest are:

         * beta=2 : Euclidean distance
         * beta=1 : Kullback Leibler
         * beta=0 : Itakura-Saito
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
            self.P = nnrandn(self.data.shape[:3] + (nb_components,))
        else:
            self.P = P

        if At is None:
            self.At = nnrandn((self.data.shape[3], nb_components))
        else:
            self.At = At

        if Ac is None:
            self.Ac = nnrandn((self.data.shape[4], nb_components))
        else:
            self.Ac = Ac

    def fit(self):
        """fits a common fate model to
        Z(a,b,f,t,i) = P(a,b,j)Af(f,j)At(t,j)Ac(i,j)
        """

        def MU(einsumString, Z, factors):
            Zhat = hat(self.P, self.At, self.Ac)
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
            self.P *= MU('abftc,tj,cj->abfj', self.data, (self.At, self.Ac))
            self.At *= MU('abftc,abfj,cj->tj', self.data, (self.P, self.Ac))
            self.Ac *= MU('abftc,abfj,tj->cj', self.data, (self.P, self.At))

        return self

    @property
    def factors(self):
        return (self.P, self.At, self.Ac)

    @property
    def approx(self):
        return hat(self.P, self.At, self.Ac)
