import numpy
import scipy
import scipy.stats
import random
from decimal import *


class cmNDIRM:
    def __init__(self):
        # dim_n: number of entities
        # dim_k: number of features
        self.alpha = 1.0
        self.mean_w = 0.0
        self.sigma_w = 2.0
        self.beta_w = 1.0
        self.alpha_hyper_parameter = (1.0, 1.0)
        self.matrix_y = numpy.zeros((0, 0))
        self.matrix_w = numpy.zeros((0, 0))
        self.matrix_z = numpy.zeros((0, 0))
        self.dim_n = 1
        self.dim_k = 1
        self.metropolis_hastings_k_new = False

        self.amount_fix = 0.01

        self.alpha_cns = 0.5
        self.beta_cns = 0.5
        self.cns = numpy.zeros((0, 0))

    def initialize_data(self, data, cns):
        self.matrix_y = data
        self.dim_n = self.matrix_y.shape[0]

        self.initialize_matrix_z()
        self.dim_k = self.matrix_z.shape[1]
        self.cns = cns #numpy.zeros((self.dim_n, self.dim_n))
        self.matrix_w = numpy.zeros((self.dim_k, self.dim_k))
        self.matrix_w = self.sample_matrix_w(self.alpha_cns, self.beta_cns, self.matrix_z, self.matrix_y, self.cns)

    def initialize_matrix_z(self):
        matrix_z = numpy.ones((0, 0))
        sample_dish = [[]]
        dim_k_new = scipy.stats.poisson.rvs((self.alpha * 1.0))

        sample_dish = numpy.hstack((sample_dish, numpy.ones((1, dim_k_new))))

        matrix_z = numpy.hstack((matrix_z, numpy.zeros((matrix_z.shape[0], dim_k_new))))
        matrix_z = numpy.vstack((matrix_z, sample_dish))
        for i in xrange(1, self.dim_n):
            matrix_z = self.sample_new_disk(i, matrix_z)

        self.dim_k = matrix_z.shape[1]
        assert (matrix_z.shape[0] == self.dim_n)
        self.matrix_z = matrix_z.astype(numpy.int)

        return self.matrix_z

    def sample_new_disk(self, i, matrix_z=None):
        if matrix_z is None:
            matrix_z = self.matrix_z

        sample_dish = (numpy.random.uniform(0, 1, (1, matrix_z.shape[1])) <
                       (matrix_z.sum(axis=0).astype(numpy.float) / i))
        dim_k_new = scipy.stats.poisson.rvs((self.alpha * 1.0 / i))

        sample_dish = numpy.hstack((sample_dish, numpy.ones((1, dim_k_new))))

        matrix_z = numpy.hstack((matrix_z, numpy.zeros((matrix_z.shape[0], dim_k_new))))
        matrix_z = numpy.vstack((matrix_z, sample_dish))

        return matrix_z

    def log_likelihood_y(self, matrix_z=None, matrix_w=None, matrix_y=None):
        if matrix_z is None:
            matrix_z = self.matrix_z
        if matrix_w is None:
            matrix_w = self.matrix_w
        if matrix_y is None:
            matrix_y = self.matrix_y

        log_likelihood = Decimal(1.0)
        dim_n = self.dim_n
        for i in range(0, dim_n):
            for j in range(0, dim_n):
                a = Decimal(self.likelihood_y_i_j(i, j, matrix_z, matrix_w, matrix_y))
                b = log_likelihood*a
                log_likelihood = Decimal(b)
        return log_likelihood

    def likelihood_y_i_j(self, i=0, j=0, matrix_z=None, matrix_w=None, matrix_y=None):
        if matrix_z is None:
            matrix_z = self.matrix_z
        if matrix_w is None:
            matrix_w = self.matrix_w
        if matrix_y is None:
            matrix_y = self.matrix_y

        dim_k = matrix_w.shape[0]

        likelihood_new = 0.0
        for k in range(0, dim_k):
            for k_prime in range(0, dim_k):
                if (matrix_z[i][k] == 1) and (matrix_z[j][k_prime] == 1):
                    likelihood_new += matrix_w[k][k_prime]
        likelihood = likelihood_new
        likelihood = 1.0/(1.0+numpy.exp(-likelihood))

        if matrix_y[i][j] == 0:
            if likelihood >= 0.5:
                likelihood = 1.0 - likelihood
        else:
            if likelihood < 0.5:
                likelihood = 1.0 - likelihood
        return likelihood

    def sample_vector_z_n(self, object_index):
        assert(type(object_index) == int or type(object_index) == numpy.int32 or type(object_index) == numpy.int64)

        # calculate initial feature possess counts
        m = self.matrix_z.sum(axis=0)

        # remove this data point from m vector
        new_m = (m - self.matrix_z[object_index, :]).astype(numpy.float)

        # compute the log probability of p(Znk=0 | Z_nk) and p(Znk=1 | Z_nk)
        log_prob_z1 = new_m / self.dim_n
        log_prob_z0 = 1.0 - (new_m / self.dim_n)

        # find all singleton features possessed by current object
        singleton_features = [nk for nk in range(self.dim_k) if self.matrix_z[object_index, nk] != 0 and new_m[nk] == 0]
        non_singleton_features = [nk for nk in range(self.dim_k) if nk not in singleton_features]

        order = numpy.random.permutation(self.dim_k)

        for (feature_counter, feature_index) in enumerate(order):
            if feature_index in non_singleton_features:
                matrix_z = numpy.copy(self.matrix_z)
                # compute the log likelihood when Znk=0
                matrix_z[object_index, feature_index] = 0
                matrix_w_0 = self.sample_matrix_w(self.alpha_cns, self.beta_cns, self.matrix_z, self.matrix_y, self.cns)
                prob_z0 = self.log_likelihood_y(matrix_z, matrix_w_0, self.matrix_y)
                prob_z0 *= Decimal(log_prob_z0[feature_index])
                #prob_z0 = numpy.exp(prob_z0)

                # compute the log likelihood when Znk=1
                matrix_z[object_index, feature_index] = 1
                matrix_w_1 = self.sample_matrix_w(self.alpha_cns, self.beta_cns, self.matrix_z, self.matrix_y, self.cns)
                prob_z1 = self.log_likelihood_y(matrix_z, matrix_w_1, self.matrix_y)
                prob_z1 *= Decimal(log_prob_z1[feature_index])
                #prob_z1 = numpy.exp(prob_z1)

                #print self.log_likelihood_y(self.matrix_z, self.matrix_w, self.matrix_y)
                #prob_z0 /= (prob_z0 + prob_z1)
                # prob_z0 = round(prob_z0.ln(), 2)
                if prob_z0 == Decimal('NaN'):
                    prob_z0 = 0
                if prob_z1 == Decimal('NaN'):
                    prob_z1 = 0
                if prob_z0 > prob_z1:
                    self.matrix_z[object_index, feature_index] = 0
                    self.matrix_w = matrix_w_0
                else:
                    self.matrix_z[object_index, feature_index] = 1
                    self.matrix_w = matrix_w_1
        return singleton_features

    def sample_matrix_z(self):
        order = numpy.random.permutation(self.dim_n)
        for (object_counter, object_index) in enumerate(order):
            # sample Z_n
            singleton_features = self.sample_vector_z_n(object_index)

            if self.metropolis_hastings_k_new:
                # sample K_new using metropolis hasting
                b = self.sample_metropolis_hastings_k_new(object_index, singleton_features)

    """
    sample K_new using metropolis hastings algorithm
    """
    def sample_metropolis_hastings_k_new(self, object_index, singleton_features):
        if type(object_index) != list:
            object_index = [object_index]

        k_temp = scipy.stats.poisson.rvs(self.alpha / self.dim_n)

        if k_temp <= 0 and len(singleton_features) <= 0:
            return False

        # generate new features from a normal distribution with mean 0 and variance sigma_a, a K_new-by-K matrix
        """
        matrix_w_new = numpy.copy(self.matrix_w)
        matrix_w_new = matrix_w_new[[k for k in xrange(self.dim_k) if k not in singleton_features], :]
        matrix_w_new = matrix_w_new[:, [k for k in xrange(self.dim_k) if k not in singleton_features]]
        """
        matrix_z_new = numpy.copy(self.matrix_z)
        """
        if k_temp != 0:
            for i in range(0, k_temp, 1):
                matrix_w_temp = numpy.random.normal(0, self.sigma_w, (1, matrix_w_new.shape[0]))
                matrix_w_new = numpy.vstack((matrix_w_new, matrix_w_temp))
                new_k_temp_k_temp = numpy.random.normal(0, self.sigma_w, (1, 1))
                matrix_w_temp = numpy.hstack((matrix_w_temp, new_k_temp_k_temp))
                matrix_w_new = numpy.hstack((matrix_w_new, matrix_w_temp.T))
        """
        # generate new z matrix row
        matrix_z_new = matrix_z_new[:, [k for k in xrange(self.dim_k) if k not in singleton_features]]
        matrix_z_new = numpy.hstack((matrix_z_new, numpy.ones((self.dim_n, k_temp))))
        matrix_z_new = matrix_z_new.astype(numpy.int)
        matrix_w_new = self.sample_matrix_w(self.alpha_cns, self.beta_cns, matrix_z_new, self.matrix_y, self.cns)
        #matrix_z_new = numpy.hstack((self.matrix_z[[object_index], [k for k in xrange(self.dim_k)
        #                            if k not in singleton_features]], numpy.ones((len(object_index), k_temp))))

        dim_k_new = self.dim_k + k_temp - len(singleton_features)

        # compute the probability of generating new features
        # prob_new = numpy.exp(self.log_likelihood_y(self.matrix_y[object_index, :], matrix_z_new, matrix_w_new))
        prob_new = self.log_likelihood_y(matrix_z_new, matrix_w_new, self.matrix_y)
        """
        # construct the A_old and Z_old
        matrix_w_old = self.matrix_w
        vector_z_old = self.matrix_z[object_index, :]
        dim_k_old = self.dim_k

        assert(matrix_w_old.shape == (dim_k_old, dim_k_old))
        assert(matrix_w_new.shape == (dim_k_new, dim_k_new))
        assert(vector_z_old.shape == (len(object_index), dim_k_old))
        assert(matrix_z_new.shape == (len(object_index), dim_k_new))
        """

        # compute the probability of using old features
        # prob_old = numpy.exp(self.log_likelihood_y(self.matrix_y[object_index, :], vector_z_old, matrix_w_old))
        prob_old = self.log_likelihood_y(self.matrix_z, self.matrix_w, self.matrix_y)

        # compute the probability of generating new features
        #prob_new /= (prob_new + prob_old)
        #prob_new = round(prob_new.ln(), 2)

        # if we accept the proposal, we will replace old W and Z matrices
        # round(prob_new.ln(), 2) > round(prob_old.ln(), 2):
        if prob_old < prob_new:
            self.matrix_w = matrix_w_new
            self.matrix_z = matrix_z_new
            #self.matrix_z = numpy.hstack((self.matrix_z[:, [k for k in xrange(self.dim_k)
            #                                                if k not in singleton_features]],
            #                              numpy.zeros((self.dim_n, k_temp))))
            #self.matrix_z[object_index, :] = matrix_z_new
            self.dim_k = dim_k_new
            return True

        return False

    def sample(self, iterations):
        vec = []
        llh = []
        for i in range(0, self.dim_k):
            vec.append([])

        for iter in xrange(iterations):
            self.sample_matrix_z()
            #self.sample_w(self.matrix_z)
            #self.sample_w_normal(self.matrix_z)
            llh.append(self.log_likelihood_y(self.matrix_z, self.matrix_w, self.matrix_y).log10())

            #self.regularize_matrices()
            #self.alpha = self.sample_alpha()
            # print iter, self.dim_k
            # print("alpha: %f\tsigma_w: %f\tmean_w: %f" % (self.alpha, self.sigma_w, self.mean_w))
            print self.matrix_z.sum(axis=0)
            aa = self.matrix_z.sum(axis=0)
            #for i in range(0, self.matrix_w.shape[0]):
            #    vec[i].append(aa[i])
        return llh

    def sample_matrix_w(self, alpha, beta, matrix_z, matrix_y, CNS):
        dim_k = self.matrix_z.shape[1]
        matrix_w_new = numpy.copy(self.matrix_w)
        for k in range(0, dim_k):
            for k_prime in range(0, dim_k):
                matrix_w_new[k][k_prime] = self.cal_w_k_k_prime(alpha, beta, k, k_prime, matrix_z, matrix_y, CNS)
        amax = numpy.amax(matrix_w_new)
        amin = numpy.amin(matrix_w_new)
        matrix_w_new = self.convert_range(matrix_w_new, (amin-amax)/2, (amax-amin)/2)
        return matrix_w_new

    def cal_w_k_k_prime(self, alpha, beta, k, k_prime, matrix_z, matrix_y, CNS):
        dim_n = self.matrix_y.shape[0]
        w_k_k_prime = 0
        for i in range(0, dim_n):
            for j in range(0, dim_n):
               if (matrix_z[i][k] == 1) and (matrix_z[j][k_prime] == 1):
                   w_k_k_prime += self.cal_psi_i_j(alpha, beta, i, j, matrix_y, CNS)
        return w_k_k_prime

    def cal_psi_i_j(self, alpha, beta, i, j, matrix_y, CNS):
        psi_i_j = alpha*matrix_y[i][j] + beta*CNS[i][j]
        return psi_i_j

    def load_lazega(self):
        f = open('../data/LazegaLawyers/ELfriend36.dat')
        data = numpy.loadtxt(f)
        return data.astype(numpy.int)

    def load_kinship(self):
        file1 = open('../data/kinship3.data')
        data = file1.readlines()

        matrix_temp = numpy.zeros((24, 24))
        for i in xrange(0, 24):
            matrix_temp[i][i] = 1
        for line in data:
            line = line.strip()
            line = line.replace(" ", "")
            row = line.split(',')
            matrix_temp[int(row[0])-1][int(row[1])-1] = 1
            matrix_temp[int(row[1])-1][int(row[0])-1] = 1
        return matrix_temp.astype(numpy.int)

    def run(self):
        """
        data = numpy.array([[1, 0, 0, 1, 1],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 0, 1],
                            [1, 1, 0, 1, 1],
                            [1, 0, 1, 1, 1]])
        """
        data = numpy.array([[1,0,1,0,1,0,0,1,1],
                            [0,1,0,0,0,0,1,0,1],
                            [1,0,1,0,0,0,1,0,1],
                            [0,0,0,1,0,0,0,1,0],
                            [1,0,0,0,1,0,1,0,1],
                            [0,0,0,0,0,1,0,1,0],
                            [0,1,1,0,1,0,1,0,0],
                            [1,0,0,1,0,1,0,1,1],
                            [1,1,1,0,1,0,0,1,1]])

        cns = numpy.array([[1,0,0.4,0,1,0,0,1,0.6],
                            [0,1,0,0,0,0,1,0,1],
                            [0.4,0,1,0,0,0,1,0,1],
                            [0,0,0,1,0,0,0,1,0],
                            [1,0,0,0,1,0,1,0,1],
                            [0,0,0,0,0,1,0,1,0],
                            [0,1,1,0,1,0,1,0,0],
                            [1,0,0,1,0,1,0,1,1],
                            [0.6,1,1,0,1,0,0,1,1]])
        #data = self.load_kinship()
        #data = self.load_lazega()
        #print data
        self.initialize_data(data, cns)
        #self.initialize_matrix_z()

        print self.matrix_z
        print self.matrix_w

        a_min = numpy.amin(self.matrix_w)
        print a_min
        llh = self.sample(10)

        #print vec
        print llh
        #print self.matrix_z

    def sigmoid(self, x):
        a = 1.0/(1.0+numpy.exp(-x))
        return a

    def convert_range(self, matrix, new_min, new_max):
        old_min = numpy.amin(matrix)
        old_max = numpy.amax(matrix)
        for i in xrange(0, matrix.shape[0]):
            for j in xrange(0, matrix.shape[1]):
                matrix[i][j] = ((matrix[i][j] - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
        return matrix

if __name__ == '__main__':
    cmNDIRM = cmNDIRM()
    cmNDIRM.run()
