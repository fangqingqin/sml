import numpy as np
from scipy.special import erf

# Functional Structure


def probability_improvement(X: np.ndarray, X_sample: np.ndarray,
                            gpr: object, xi: float = 0.01) -> np.ndarray:
    """
    Probability improvement acquisition function.

    Computes the PI at points X based on existing samples X_sample using
    a Gaussian process surrogate model

    Arguments:
    ----------
        X: ndarray of shape (m, d)
            The point/points for which the expected improvement needs to be computed.

        X_sample: ndarray of shape (n, d)
            Sample locations

        gpr: GPR object.
            Gaussian process trained on previously evaluated hyperparameters.

        xi: float. Default 0.01
            Exploitation-exploration trade-off parameter.

    Returns:
    --------
        PI: ndarray of shape (m,)
    """
    # TODO Q2.5
    # Implement the probability of improvement acquisition function

    # Predict mean and standard deviation at points X
    mu, sigma = gpr.predict(X, return_std=True)

    # Current best observation
    mu_sample = gpr.predict(X_sample)

    # Get the maximum value of the current best observation
    mu_sample_opt = np.max(mu_sample)
    
    mu = mu.ravel()
    sigma = sigma.ravel()
    # sigma = sigma.reshape(-1, 1)
    
    # Calculate the improvement
    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        PI = 0.5 * (1 + erf(Z / np.sqrt(2)))
    
    return PI


def expected_improvement(X: np.ndarray, X_sample: np.ndarray,
                         gpr: object, xi: float = 0.01) -> np.ndarray:
    """
    Expected improvement acquisition function.

    Computes the EI at points X based on existing samples X_sample using
    a Gaussian process surrogate model

    Arguments:
    ----------
        X: ndarray of shape (m, d)
            The point/points for which the expected improvement needs to be computed.

        X_sample: ndarray of shape (n, d)
            Sample locations

        gpr: GPR object.
            Gaussian process trained on previously evaluated hyperparameters.

        xi: float. Default 0.01
            Exploitation-exploration trade-off parameter.

    Returns:
    --------
        EI : ndarray of shape (m,)
    """

    # TODO Q2.5
    # Implement the expected improvement acquisition function

    # Predict mean and standard deviation at points X
    mu, sigma = gpr.predict(X, return_std=True)
    mu = mu.ravel()
    sigma = sigma.ravel()

    # Current best observation
    mu_sample = gpr.predict(X_sample)
    # normalize the mu_sample
    mu_sample_opt = np.max(mu_sample)
    
    # Calculate the improvement
    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        cdf_Z = 0.5 * (1 + erf(Z / np.sqrt(2)))
        pdf_Z = np.exp(-0.5 * Z**2) / np.sqrt(2 * np.pi)
        ei = imp * cdf_Z + sigma * pdf_Z
        ei[sigma == 0.0] = 0.0
    
    return ei