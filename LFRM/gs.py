import abc
import numpy
import scipy
import math


class GibbsSampling(object):
    __metaclass__ = abc.ABCMeta

    """
    @param gibbs_sampling_maximum_iteration: gibbs sampling maximum iteration
    @param alpha_hyper_parameter: hyper-parameter for alpha sampling, a tuple defining the parameter for an inverse gamma distribution
    @param sigma_a_hyper_parameter: hyper-parameter for sigma_a sampling, a tuple defining the parameter for an inverse gamma distribution
    @param sigma_x_hyper_parameter: hyper-parameter for sigma_x sampling, a tuple defining the parameter for an inverse gamma distribution
    @param metropolis_hasting_k_new: a boolean variable, set to true if we use metropolis hasting to estimate K_new, otherwise use truncated gibbs sampling
    @param snapshot_interval: the interval for exporting a snapshot of the model
    """

    def __init__(self,  # real_valued_latent_feature=True,
                 alpha_hyper_parameter=None,
                 sigma_w_hyper_parameter=None,
                 sigma_y_hyper_parameter=None,
                 metropolis_hastings_k_new=True,
                 snapshot_interval=10):
        # initialize the hyper-parameter for sampling _alpha
        # a value of None is a gentle way to say "do not sampling _alpha"
        assert (alpha_hyper_parameter is None or type(alpha_hyper_parameter) == tuple)
        self._alpha_hyper_parameter = alpha_hyper_parameter
        # initialize the hyper-parameter for sampling _sigma_x
        # a value of None is a gentle way to say "do not sampling _sigma_x"
        assert (sigma_y_hyper_parameter is None or type(sigma_y_hyper_parameter) == tuple)
        self._sigma_y_hyper_parameter = sigma_y_hyper_parameter
        # initialize the hyper-parameter for sampling _sigma_a
        # a value of None is a gentle way to say "do not sampling _sigma_a"
        assert (sigma_w_hyper_parameter is None or type(sigma_w_hyper_parameter) == tuple)
        self._sigma_w_hyper_parameter = sigma_w_hyper_parameter

        self._alpha = 1.0

        # self._real_valued_latent_feature = real_valued_latent_feature
        self._metropolis_hastings_k_new = metropolis_hastings_k_new

        self._snapshot_interval = snapshot_interval

        self._y_title = "Y-matrix-"
        self._z_title = "Z-matrix-"
        self._w_title = "W-matrix-"
        self._hyper_parameter_vector_title = "Hyper-parameter-vector-"

    """
    @param data: a NxD NumPy data matrix
    @param alpha: IBP hyper parameter
    @param sigma_x: standard derivation of the noise on data, often referred as sigma_n as well
    @param sigma_a: standard derivation of the feature, often referred as sigma_f as well
    @param initializ_Z: seeded Z matrix
    """

    @abc.abstractmethod
    def _initialize(self, data, alpha=1.0, sigma_w=1.0, sigma_y=1.0, W_prior=None, initial_Z=None):
        self._alpha = alpha
        self._sigma_y = sigma_y
        self._sigma_w = sigma_w

        # Data matrix
        # self._X = self.center_data(data)
        self._Y = data
        (self._N, self._N) = self._Y.shape

        if initial_Z is None:
            # initialize Z from IBP(alpha)
            self._Z = self.initialize_Z()
        else:
            self._Z = initial_Z

        assert (self._Z.shape[0] == self._N)

        print self._Z

        # make sure Z matrix is a binary matrix
        # assert (self._Z.dtype == numpy.int)
        assert (self._Z.max() == 1 and self._Z.min() == 0)

        # record down the number of features
        self._K = self._Z.shape[1]

        if W_prior is None:
            self._W_prior = numpy.zeros((1, self._N))
        else:
            self._W_prior = W_prior
        assert (self._W_prior.shape == (1, self._N))

        self._W = self.initialize_W()
        assert (self._W.shape == (self._K, self._K))

        return

    """
    initialize latent feature appearance matrix Z according to IBP(alpha)
    """

    def initialize_Z(self):
        Z = numpy.ones((0, 0))
        # initialize matrix Z recursively in IBP manner
        for i in xrange(1, self._N + 1):
            # sample existing features
            # Z.sum(axis=0)/i: compute the popularity of every dish, computes the probability of sampling that dish
            sample_dish = (numpy.random.uniform(0, 1, (1, Z.shape[1])) < (Z.sum(axis=0).astype(numpy.float) / i))
            # sample a value from the poisson distribution, defines the number of new features
            K_new = scipy.stats.poisson.rvs((self._alpha * 1.0 / i))
            # horizontally stack or append the new dishes to current object's observation vector, i.e., the vector Z_{n*}
            sample_dish = numpy.hstack((sample_dish, numpy.ones((1, K_new))))
            # append the matrix horizontally and then vertically to the Z matrix
            Z = numpy.hstack((Z, numpy.zeros((Z.shape[0], K_new))))
            Z = numpy.vstack((Z, sample_dish))

        assert (Z.shape[0] == self._N)
        Z = Z.astype(numpy.float)

        return Z

    """
    maximum a posterior estimation of matrix A
    todo: 2D-prior on A when initializing A matrix
    """

    def initialize_W(self):
        W_new = numpy.random.normal(0, self._sigma_w, (self._K, self._K))
        assert (W_new.shape == (self._K, self._K))

        return W_new

    """
    sample standard deviation of a multivariant Gaussian distribution
    @param sigma_hyper_parameter: the hyper-parameter of the gamma distribution
    @param matrix: a r*c matrix drawn from a multivariant c-dimensional Gaussian distribution with zero mean and identity c*c covariance matrix
    """

    @staticmethod
    def sample_sigma(sigma_hyper_parameter, matrix):
        assert (sigma_hyper_parameter != None)
        assert (matrix != None)
        assert (type(sigma_hyper_parameter) == tuple)
        assert (type(matrix) == numpy.ndarray)

        (sigma_hyper_a, sigma_hyper_b) = sigma_hyper_parameter
        (row, column) = matrix.shape

        # compute the posterior_shape = sigma_hyper_a + n/2, where n = self._D * self._K
        posterior_shape = sigma_hyper_a + 0.5 * row * column
        # compute the posterior_scale = sigma_hyper_b + sum_{k} (A_k - \mu_A)(A_k - \mu_A)^\top/2
        var = 0
        if row >= column:
            var = numpy.trace(numpy.dot(matrix.transpose(), matrix))
        else:
            var = numpy.trace(numpy.dot(matrix, matrix.transpose()))

        posterior_scale = 1.0 / (sigma_hyper_b + var * 0.5)
        tau = scipy.stats.gamma.rvs(posterior_shape, scale=posterior_scale)
        sigma_w_new = numpy.sqrt(1.0 / tau)

        return sigma_w_new

    """
    sample alpha from conjugate posterior
    """

    def sample_alpha(self):
        assert (self._alpha_hyper_parameter != None)
        assert (type(self._alpha_hyper_parameter) == tuple)

        (alpha_hyper_a, alpha_hyper_b) = self._alpha_hyper_parameter

        posterior_shape = alpha_hyper_a + self._K
        H_N = numpy.array([range(self._N)]) + 1.0
        H_N = numpy.sum(1.0 / H_N)
        posterior_scale = 1.0 / (alpha_hyper_b + H_N)

        # posterior_shape = alpha_hyper_a + self._Z.sum()
        # posterior_scale = 1.0/(alpha_hyper_b + self._N)

        alpha_new = scipy.stats.gamma.rvs(posterior_shape, scale=posterior_scale)

        return alpha_new

    """
    compute the log-likelihood of the Z matrix.
    """

    def log_likelihood_Z(self):
        # compute {K_+} \log{\alpha} - \alpha * H_N, where H_N = \sum_{j=1}^N 1/j
        H_N = numpy.array([range(self._N)]) + 1.0
        H_N = numpy.sum(1.0 / H_N)
        log_likelihood = self._K * numpy.log(self._alpha) - self._alpha * H_N

        # compute the \sum_{h=1}^{2^N-1} \log{K_h!}
        Z_h = numpy.sum(self._Z, axis=0).astype(numpy.int)
        Z_h = list(Z_h)
        for k_h in set(Z_h):
            log_likelihood -= numpy.log(math.factorial(Z_h.count(k_h)))

        # compute the \sum_{k=1}^{K_+} \frac{(N-m_k)! (m_k-1)!}{N!}
        for k in xrange(self._K):
            m_k = Z_h[k]
            temp_var = 1.0
            if m_k - 1 < self._N - m_k:
                for k_prime in range(self._N - m_k + 1, self._N + 1):
                    if m_k != 1:
                        m_k -= 1

                    temp_var /= k_prime
                    temp_var *= m_k
            else:
                n_m_k = self._N - m_k
                for k_prime in range(m_k, self._N + 1):
                    temp_var /= k_prime
                    temp_var += n_m_k
                    if n_m_k != 1:
                        n_m_k -= 1

            log_likelihood += numpy.log(temp_var)

        return log_likelihood

    """
    @param directory: the export directory
    @param index: the export index, e.g., usually the iteration count, append to the title
    """

    def export_snapshot(self, directory, index):
        import os
        if not os.path.exists(directory):
            os.mkdir(directory)
        assert (directory.endswith("/"))
        numpy.savetxt(directory + self._w_title + str(index), self._W)
        numpy.savetxt(directory + self._y_title + str(index), self._Y)
        numpy.savetxt(directory + self._z_title + str(index), self._Z)
        vector = numpy.array([self._alpha, self._sigma_w, self._sigma_y])
        numpy.savetxt(directory + self._hyper_parameter_vector_title + str(index), vector)
        print "successfully export the snapshot to " + directory + " for iteration " + str(index) + "..."

    """
    @param directory: the import director
    @param index: the import index, e.g., usually the iteration count, append to the title
    """

    def import_snapshot(self, directory, index):
        assert (directory.endswith("/"))
        self._W = numpy.loadtxt(directory + self._w_title + str(index))
        self._Y = numpy.loadtxt(directory + self._y_title + str(index))
        self._Z = numpy.loadtxt(directory + self._z_title + str(index))
        (self._N, self._K) = self._Z.shape
        (self._N, self._N) = self._Y.shape
        assert (self._Z.shape[0] == self._y.shape[0])
        assert (self._W.shape == (self._K, self._K))
        (self._alpha, self._sigma_w, self._sigma_y) = numpy.loadtxt(
            directory + self._hyper_parameter_vector_title + str(index))
        print "successfully import the snapshot from " + directory + " for iteration " + str(index) + "..."

    """
    center the data, i.e., subtract the mean
    """

    @staticmethod
    def center_data(data):
        (N, N) = data.shape
        data = data - numpy.tile(data.mean(axis=0), (N, 1))
        return data
