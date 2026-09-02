import itertools
import pytest
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from petsc4py import PETSc
import pymgmc


class MarginalisedDistribution:
    """Marginalised distribution

    The full distribution is

    .. math::

        p(y_1,y_2) \\propto \\frac{\\Lambda(y_1)^n}{n!} e^{-\\Lambda(y_1)}
        \\exp\\left[-\\frac{1}{2}y^\\top A y + f^\\top y\\right]

    for SPD :math:`2\\times 2` pecision matrix :math:`A` and right hand
    side vector :math:`f`. The rate is given by
    :math:`\\Lambda(y) = e^{b y +\\nu}`. Marginalising over :math:`y_2`
    gives

    .. math::
        p_{mar}(y_1) \\propto \\frac{\\Lambda(y_1)^n}{n!} e^{-\\Lambda(y_1)}
        \\exp\\left[-\\frac{1}{2} a_{mar} y_1^2 + f_{mar} y_1 \\right]

    with :math:`a_{mar} = A_{11} - \\frac{A_{12}^2}{A_{22}}` and
    :math:`f_{mar} = f_1 - \\frac{A_{12}}{A_{22}} f_2`.
    """

    # Lower and upper bounds for integrals
    LOWER_BOUND = -100
    UPPER_BOUND = +100

    def __init__(self, Q_prec, f_rhs, b_measurement, nu, event_count):
        """Initialise new instance

        Parameters
        ==========
        Q_prec :
            precision matrix :math:`Q`
        f_rhs :
            right hand side vector :math:`f`
        b_measurement :
            strength of measurement :math:`b`
        nu :
            offset :math:`nu`
        event_count :
            measured number of events :math:`n`
        """
        self._Q_prec = Q_prec
        self._f_rhs = f_rhs
        self._b_measurement = b_measurement
        self._nu = nu
        self._event_count = event_count
        # normalise marginal distribution
        self._Znorm = (
            1
            / sp.integrate.quad(
                self._unnormalised_pdf,
                MarginalisedDistribution.LOWER_BOUND,
                MarginalisedDistribution.UPPER_BOUND,
            )[0]
        )

    def _unnormalised_pdf(self, y):
        """Unnormalised PDF

        Computes :math:`Z p_{mar}(y)` with unknown normalisation factor
        :math:`Z`.

        Parameters
        ==========
        y : float
            point at which to evaluate the probability density

        Returns
        =======
        Value of unnormalised PDF :math:`Z p_{mar}(y)`
        """
        Lambda = np.exp(self._b_measurement * y + self._nu)
        p_poisson = Lambda**self._event_count * np.exp(-Lambda)
        p_marginal = np.exp(
            -1
            / 2
            * (self._Q_prec[0, 0] - self._Q_prec[0, 1] ** 2 / self._Q_prec[1, 1])
            * y**2
            + (
                self._f_rhs[0]
                - self._Q_prec[0, 1] / self._Q_prec[1, 1] * self._f_rhs[1]
            )
            * y
        )
        return p_poisson * p_marginal

    def pdf(self, y):
        """Normalised PDF

        Computes normalised :math:`p_{mar}(y)`

        Parameters
        ==========
        y : float
            point at which to evaluate the probability density

        Returns
        =======
        Value of PDF :math:`p_{mar}(y)`
        """
        return self._Znorm * self._unnormalised_pdf(y)

    def cdf(self, y):
        """CDF :math:`\\int_{-\\infty}^y p_{mar}(z)\\;dz`

        Parameters
        ==========
        y : float
            point at which to evaluate the CDF

        Returns
        =======
        Value of CDF :math:`\\int_{-\\infty}^y p_{mar}(z)\\;dz`
        """
        return np.vectorize(
            lambda z: sp.integrate.quad(
                self.pdf, MarginalisedDistribution.LOWER_BOUND, z
            )[0]
        )(y)

    def kolmogorov_smirnov(self, samples):
        """Kolmogorov Smirnov test

        Computes the maximum of the Kolmogorov Smirnov statistic

        .. math::

            \\max_{x\\in S} \\|CDF(x)-\\widehat{CDF}(x)\\|

        for a given set of samples :math:`S=\\{x_j\\}_{j=1}^{N}`. After
        ordering the samples in increasing order, this is approximated by

        .. math::

            \\max_{j=1,2,\\dots,N} \\{\\|CDF(x_j)- \\frac{j}{N}\\|,\\|CDF(x_j)- \\frac{j-1}{N}\\|\\}

        Parameters
        ==========
        samples :
            list of samples :math:`S=\\{x_j\\}_{j=1}^{N}`

        Returns
        =======
        Value of Kolmogorov Smirnov statistic
        """
        CDF_true = self.cdf(np.asarray(sorted(samples)))
        n_samples = len(samples)
        CDF_empirical_left = np.arange(n_samples) / n_samples
        CDF_empirical_right = (1 + np.arange(n_samples)) / n_samples
        D_left = np.max(np.abs(CDF_empirical_left - CDF_true))
        D_right = np.max(np.abs(CDF_empirical_right - CDF_true))
        return max(D_left, D_right)


class Sampler:
    """Wrapper for sampler for 2 distribution

    The distribution is given by

    .. math::

        p(y_1,y_2) \\propto \\frac{\\Lambda(y_1)^n}{n!} e^{-\\Lambda(y_1)}
        \\exp\\left[-\\frac{1}{2}y^\\top A y + f^\\top y\\right]

    for SPD :math:`2\\times 2` pecision matrix :math:`A` and right hand
    side vector :math:`f`. The rate is given by
    :math:`\\Lambda(y) = e^{b y +\\nu}`.
    """

    def __init__(self, Q, f, b, nu, event_count):
        """Initialise new instance

        Parameters
        ==========
        Q :
            precision matrix :math:`Q`
        f_rhs :
            right hand side vector :math:`f`
        b_measurement :
            strength of measurement :math:`b`
        nu :
            offset :math:`nu`
        event_count :
            measured number of events :math:`n`
        """
        # PETSc objects
        Q_petsc = PETSc.Mat().createAIJWithArrays(
            (2, 2), ((0, 2, 4), (0, 1, 0, 1), [Q[0, 0], Q[0, 1], Q[1, 0], Q[1, 1]])
        )
        B_petsc = PETSc.Mat().createAIJWithArrays((2, 1), ((0, 1, 2), (0, 0), [b, 0]))

        f_petsc = PETSc.Vec().createWithArray(f)
        nu_petsc = PETSc.Vec().createWithArray([nu])
        event_count_petsc = PETSc.Vec().createWithArray([event_count])
        snes = PETSc.SNES().create()
        opts = PETSc.Options()
        solver_parameters = {
            "snes_type": "poissongibbs",
            "poissongibbs_its": 1,
            "snes_view": ":snes_view.txt",
        }
        for key, value in solver_parameters.items():
            opts[key] = value
        snes.setFromOptions()

        pymgmc.SNESPoissonSetAppCtx(
            snes, event_count_petsc, Q_petsc, B_petsc, f_petsc, nu_petsc
        )
        self._snes = snes
        self._y = PETSc.Vec().createWithArray([0, 0])

    def __iter__(self):
        """Iterator"""
        while True:
            self._snes.solve(None, self._y)
            yield np.array(self._y.getArray())


def visualise(
    Q_prec,
    f_rhs,
    b_measurement,
    nu,
    event_count,
    n_samples=10000,
    n_bins=128,
    y_min=-3.0,
    y_max=2.0,
    filename="marginal.pdf",
):
    """Plot PDF and historgram of samples

    Parameters
    ==========
    Q_prec :
        precision matrix :math:`Q`
    f_rhs :
        right hand side vector :math:`f`
    b_measurement :
        strength of measurement :math:`b`
    nu :
        offset :math:`nu`
    event_count :
        measured number of events :math:`n`
    n_samples :
        number of samples to use
    n_bins :
        number of bins
    y_min :
        lower bound for plotting
    y_max :
        upper bound for plotting
    filename :
        name of file to save to
    """
    distribution = MarginalisedDistribution(
        Q_prec, f_rhs, b_measurement, nu, event_count
    )
    sampler = Sampler(Q_prec, f_rhs, b_measurement, nu, event_count)
    samples = np.asarray(list(itertools.islice(sampler, n_samples)))[:, 0]
    ks = distribution.kolmogorov_smirnov(samples)
    print(f"Kolmgorov-Smirnow statistics = {ks:6.2e}")

    plt.clf()
    ax = plt.gca()
    ax.set_xlabel("$y$")
    X = np.arange(y_min, y_max, 0.01)
    plt.plot(X, distribution.pdf(X), label="exact PDF")
    plt.hist(samples, bins=n_bins, range=(y_min, y_max), density=True, label="samples")
    plt.legend(loc="upper left")
    plt.savefig(filename, bbox_inches="tight")


# Configurations used for testing
test_data = [
    (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0.0, 0.1]), 1.0, 0.0, 1),
    (np.array([[1.0, -0.1], [-0.1, 2.0]]), np.array([0.2, 0.8]), 2.0, 0.5, 2),
    (np.array([[1.0, -0.1], [-0.1, 2.0]]), np.array([0.2, 0.8]), 2.0, 0.5, 2),
    (np.array([[1.0, 0.5], [0.5, 2.0]]), np.array([0.2, 0.8]), 2.0, 0.5, 2),
    (np.array([[1.0, 0.5], [0.5, 2.0]]), np.array([0.2, 0.8]), 1.0, 0.0, 4),
]


@pytest.mark.parametrize("Q_prec,f_rhs,b_measurement,nu,event_count", test_data)
def test_sampler(Q_prec, f_rhs, b_measurement, nu, event_count):
    """Verify that samples satisfy the Kolmogorov Smirnov test"""
    n_samples = 20000
    distribution = MarginalisedDistribution(
        Q_prec, f_rhs, b_measurement, nu, event_count
    )
    sampler = Sampler(Q_prec, f_rhs, b_measurement, nu, event_count)
    samples = np.asarray(list(itertools.islice(sampler, n_samples)))[:, 0]
    tolerance = 1.0e-2
    assert distribution.kolmogorov_smirnov(samples) < tolerance


if __name__ == "__main__":
    A_precision = np.array([[1.0, -0.1], [-0.1, 2.0]])
    f_rhs = np.array([0.2, 0.8])
    b_measurement = 2.0
    nu = 0.5
    event_count = 1
    visualise(A_precision, f_rhs, b_measurement, nu, event_count)
