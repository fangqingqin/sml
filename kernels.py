import numpy as np

# Class Structure


class Matern:
    """
    Matern kernel.

    Arguments:
    ----------
    nu : float
        The parameter nu controlling the smoothness of the learned function.

    length_scale : float, default=1.0
        The length scale of the kernel.

    length_scale_bounds : pair of floats >= 0, default=(1e-5, 1e3)
        The lower and upper bound on 'length_scale'.

    variance : float, default=1.0
        The signal variance of the kernel

    variance_bounds : pair of floats >= 0, default=(1e-5, 1e2)
        The lower and upper bound on 'variance'.
    """

    def __init__(self, nu: float, length_scale: float = 1.0, length_scale_bounds: tuple = (1e-5, 1e3),
                 variance: float = 1.0, variance_bounds: tuple = (1e-5, 1e2)) -> None:
        self.nu = nu
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.variance = variance
        self.variance_bounds = variance_bounds

        self.hyperparameters = [{'name': 'length_scale', 'value': length_scale, 'bounds': self.length_scale_bounds},
                                {'name': 'variance', 'value': variance, 'bounds': self.variance_bounds}]
        
        for hp in self.hyperparameters:
            setattr(self, hp['name'], hp['value'])
            setattr(self, f"{hp['name']}_bounds", hp['bounds'])

    def get_theta(self):
        """
        Returns the log-transformed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns:
        --------
        theta : ndarray of shape (n_dims,)
            The log-transformed hyperparameters of the kernel
        """
        
        theta = [getattr(self, hp['name']) for hp in self.hyperparameters]
        return np.hstack(theta)

    def set_theta(self, theta):
        hyperparams = self.get_hyperparameters()
        
        for i in range(len(self.hyperparameters)):
           hyperparams[self.hyperparameters[i]['name']] = theta[i]
        # Update set_hyperparameters
        self.set_hyperparameters(**hyperparams)

    def get_bounds(self):
        bounds = [getattr(self, f"{hp['name']}_bounds") for hp in self.hyperparameters]
        return np.vstack(bounds)

    def get_hyperparameters(self):
        return {hp['name']: getattr(self, hp['name']) for hp in self.hyperparameters}

    def set_hyperparameters(self, **hyperparams):
        for hp in self.hyperparameters:
            if hp['name'] in hyperparams:
                setattr(self, hp['name'], hyperparams[hp['name']])



    def __call__(self, X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
        """
        Return the kernel k(X, Y).

        Arguments:
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            should be evaluated instead.

        Returns:
        --------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        """

        X = np.atleast_2d(X)
        length_scale = np.squeeze(self.length_scale).astype(float)

        # TODO Q2.2b
        # Uncomment the code and implement the Matern class covariance functions for different values of nu
        # FIXME
        if Y is None:
            Y = X
        else:
            Y = np.atleast_2d(Y)

        # self.length_scale = ell, dists = d/ell
        dists = np.linalg.norm(X[:, np.newaxis] - Y[np.newaxis, :], axis=2) / length_scale

        # k_{\text{Matérn}}(x, y) = \sigma_f^2 \cdot \exp\left(-\frac{\sqrt{2(p+\frac{1}{2}) d}}{\ell}\right) 
        # \cdot \frac{\Gamma\left(p + 1\right)}{\Gamma\left(2p + 1\right)} 
        # \cdot \sum_{i=0}^{p} \frac{(p + i)!}{i! (p-i)!} \left(2\frac{\sqrt{2(p+\frac{1}{2}) d}}{\ell}\right)^{p-i}
        # self.variance = sigma_f^2, 
        # nu = 0.5, 1.5, 2.5, p = 0, 1, 2 respectively
        if self.nu == 0.5:
            # p = 0, 2p + 1 = 1, Gamma(1)/Gamma(1) = 1,
            # i = 0, (p + i)! = 1, i! = 1, (p-i)! = 1
            K = self.variance * np.exp(-dists)
        elif self.nu == 1.5:
            # p = 1, 2p + 1 = 3, Gamma(2)/Gamma(3) = 1/2,  2*sqrt(2p+1) = 2*sqrt(3)
            # i = 0, (p + i)! = 1, i! = 1, (p - i)! = 1, p - i = 1
            # i = 1, (p + i)! = 2, i! = 1, (p - i)! = 1, p - i = 0
            K = self.variance * np.exp(-np.sqrt(3) * dists) * (np.sqrt(3) * dists + 1)
        elif self.nu == 2.5:
            # p = 2, 2p + 1 = 5, Gamma(3)/Gamma(5) = 2/24, 2*sqrt(2p+1) = 2*sqrt(5)
            # i = 0, (p + i)! = 2, i! = 1, (p - i)! = 2, p - i = 2
            # i = 1, (p + i)! = 6, i! = 1, (p - i)! = 1, p - i = 1
            # i = 2, (p + i)! = 24, i! = 2, (p - i)! = 1, p - i = 0
            K = self.variance * np.exp(-np.sqrt(5) * dists) * (5/3 * dists**2 + np.sqrt(5) * dists + 1)

        return K

