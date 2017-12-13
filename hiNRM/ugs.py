"""
@author: Ke Zhai (zhaike@cs.umd.edu)

This code was modified from the code originally written by David Andrzejewski (david.andrzej@gmail.com).
Implements uncollapsed Gibbs sampling for the linear-Gaussian infinite latent feature model (IBP).
"""

import numpy
import scipy
import random
import scipy.stats
from sklearn import preprocessing
from hiNRM.gs import GibbsSampling


# We will be taking log(0) = -Inf, so turn off this warning
numpy.seterr(divide='ignore')

class UncollapsedGibbsSampling(GibbsSampling):
    """
    @param data: a NxD NumPy data matrix
    @param alpha: IBP hyper parameter
    @param sigma_x: standard derivation of the noise on data, often referred as sigma_n as well
    @param sigma_a: standard derivation of the feature, often referred as sigma_f as well
    @param initializ_Z: seeded Z matrix
    """
    def _initialize(self, data, alpha=1.0, sigma_f=1.0, sigma_x=1.0, initial_Z=None, W_prior=None, initial_W=None):
        # Data matrix
        super(UncollapsedGibbsSampling, self)._initialize(data, alpha, sigma_f, sigma_x, W_prior, initial_Z)

        if initial_W is not None:
            # this will replace the A matrix generated in the super class. 
            self._W = initial_W
        self._W = self.initialize_W()
        assert(self._W.shape == (self._K, self._K))
    
    """
    sample the corpus to train the parameters
    """
    def sample(self, iteration, directory="../output/lfrm-output/"):
        import os
        import shutil
        if os.path.exists(directory):
            shutil.rmtree(directory)
            
        assert(self._Z.shape == (self._N, self._K))
        assert(self._W.shape == (self._K, self._K))
        assert(self._Y.shape == (self._N, self._N))
        
        #sample the total data
        for iter in xrange(iteration):
            # sample every object
            order = numpy.random.permutation(self._N)
            for (object_counter, object_index) in enumerate(order):
                # sample Z_n
                singleton_features = self.sample_Zn(object_index)

                if self._metropolis_hastings_k_new:
                    # sample K_new using metropolis hasting
                    self.metropolis_hastings_K_new(object_index)
                
            # regularize matrices
            self.regularize_matrices()
            self.sample_W(self._Z)
            # self._W = self.sample_W(self._K)
            
            #if self._alpha_hyper_parameter is not None:
            #    self._alpha = self.sample_alpha()
            
            #if self._sigma_y_hyper_parameter is not None:
            #    self._sigma_y = self.sample_sigma_y(self._sigma_y_hyper_parameter)
            
            #if self._sigma_w_hyper_parameter is not None:
            #    self._sigma_w = self.sample_sigma_w(self._sigma_w_hyper_parameter)
                
            print("iteration: %i\tK: %i\tlikelihood: %f" % (iter, self._K, self.log_likelihood_model()))
            # print("alpha: %f\tsigma_a: %f\tsigma_x: %f" % (self._alpha, self._sigma_w, self._sigma_y))
            print self._Z.sum(axis=0)
            if (iter + 1) % self._snapshot_interval == 0:
                self.export_snapshot(directory, iter + 1)
          
    """
    @param object_index: an int data type, indicates the object index (row index) of Z we want to sample
    """
    def sample_Zn(self, object_index):
        assert(type(object_index) == int or type(object_index) == numpy.int32 or type(object_index) == numpy.int64)
        
        # calculate initial feature possess counts
        m = self._Z.sum(axis=0)
        
        # remove this data point from m vector
        new_m = (m - self._Z[object_index, :]).astype(numpy.float)
        
        # compute the log probability of p(Znk=0 | Z_nk) and p(Znk=1 | Z_nk)
        log_prob_z1 = numpy.log(1.0 * new_m / self._N)
        log_prob_z0 = numpy.log(1.0 - new_m / self._N)
        
        # find all singleton features possessed by current object
        singleton_features = [nk for nk in range(self._K) if self._Z[object_index, nk] != 0 and new_m[nk] == 0]
        non_singleton_features = [nk for nk in range(self._K) if nk not in singleton_features]
        
        order = numpy.random.permutation(self._K)
        for (feature_counter, feature_index) in enumerate(order):
            if feature_index in non_singleton_features:
                #old_Znk = self._Z[object_index, feature_index]


                # compute the log likelihood when Znk=0
                self._Z[object_index, feature_index] = 0
                #w0 = self.sample_W(self._Z)
                prob_z0 = self.log_likelihood_Y(None, self._Z, self._W) #(self._Y[[object_index], :], self._Z[[object_index], :])
                prob_z0 += log_prob_z0[feature_index]
                prob_z0 = numpy.exp(prob_z0)

                # compute the log likelihood when Znk=1
                self._Z[object_index, feature_index] = 1
                #w1 = self.sample_W(self._Z)

                prob_z1 = self.log_likelihood_Y(None, self._Z, self._W) #(self._Y[[object_index], :], self._Z[[object_index], :])
                prob_z1 += log_prob_z1[feature_index]
                prob_z1 = numpy.exp(prob_z1)

                Znk_is_0 = prob_z0 / (prob_z0 + prob_z1)

                if random.random() < Znk_is_0:
                    self._Z[object_index, feature_index] = 0
                    #self._W = numpy.copy(w0)
                else:
                    self._Z[object_index, feature_index] = 1
                    #self._W = numpy.copy(w1)
                    
        return singleton_features

    """
    sample K_new using metropolis hastings algorithm
    """
    def metropolis_hastings_K_new(self, object_index):
        #if type(object_index) != list:
        #    object_index = [object_index]
    
        # sample K_new from the metropolis hastings proposal distribution, i.e., a poisson distribution with mean \frac{\alpha}{N}
        K_temp = scipy.stats.poisson.rvs(self._alpha / self._N)
        
        if K_temp <= 0:
            return False


        # generate new features from a normal distribution with mean 0 and variance sigma_w, a K_new-by-D matrix
        #W_temp = numpy.random.normal(0, self._sigma_w, (K_temp, self._K))
        #W_new = numpy.vstack((self._W, W_temp))
        #W_temp = numpy.vstack((W_temp.transpose(), numpy.random.normal(0, self._sigma_w, (K_temp, K_temp))))

        #W_new = numpy.hstack((W_new, W_temp))
        # generate new z matrix row
        #print K_temp, object_index, [k for k in xrange(self._K) if k not in singleton_features], self._Z[[object_index], [k for k in xrange(self._K) if k not in singleton_features]].shape, numpy.ones((len(object_index), K_temp)).shape

        
        # construct the A_old and Z_old
        W_old = numpy.copy(self._W)
        Z_old = numpy.copy(self._Z)
        K_old = self._K

        # compute the probability of using old features
        prob_old = numpy.exp(self.log_likelihood_Y(self._Y, Z_old, W_old))


        Z_new = numpy.hstack((self._Z, numpy.zeros((self._N, K_temp))))

        K_new = self._K + K_temp

        for i in range(1, K_temp):
            Z_new[object_index][self._K + i] = 1

        #self.sample_W(Z_new)
        W_temp = numpy.random.normal(0, self._sigma_w, (K_temp, self._K))
        W_new = numpy.vstack((self._W, W_temp))
        W_temp = numpy.vstack((W_temp.transpose(), numpy.random.normal(0, self._sigma_w, (K_temp, K_temp))))

        W_new = numpy.hstack((W_new, W_temp))

        # compute the probability of generating new features
        prob_new = numpy.exp(self.log_likelihood_Y(self._Y, Z_new, W_new))
        assert(W_old.shape == (K_old, self._K))
        assert(W_new.shape == (K_new, K_new))
        assert(Z_old.shape == (self._N, K_old))
        assert(Z_new.shape == (self._N, K_new))

        # compute the probability of generating new features
        prob_new = prob_new / (prob_old + prob_new)

        # if we accept the proposal, we will replace old A and Z matrices
        if random.random() < prob_new:
            # construct A_new and Z_new
            self._W = numpy.copy(W_new)
            #self._Z = numpy.hstack((self._Z, numpy.zeros((self._N, K_temp))))
            self._Z = numpy.copy(Z_new)
            self._K = K_new
            return True

        return False

    """
    Metropolis-Hasting sample W
    """
    def sampleMH_W(self, Z):
        print "MH W"

    def sample_W(self, Z):
        #W_old = numpy.copy(self._W)
        #W_new = numpy.random.normal(numpy.mean(self._W), self._sigma_w, (self._K, self._K))
        #W_new = numpy.zeros((Z.shape[1], Z.shape[1]))
        #W_new = numpy.copy(self._W)
        W_old = numpy.copy(self._W)
        W_new = numpy.copy(self._W)
        for k in range(Z.shape[1]):
            for k_prime in range(k, Z.shape[1]):
                min_a = min(Z[:, k].sum(axis=0), Z[:, k_prime].sum(axis=0))
                #if k == k_prime:
                a = numpy.dot(Z[:, k], Z[:, k_prime]) / Z.shape[0]
                #print a, Z.shape[0]
                #b = numpy.random.beta(1.0 + a, Z.shape[0] - a - 1.0, 1)[0]
                b = numpy.random.binomial(Z.shape[0], a, 1)[0]
                logit_s =  numpy.random.normal(a, self._sigma_w) #self.logit(b) #
                W_new[k][k_prime] = logit_s #1.0 * self.cal_w_k_k_prime(k, k_prime, Z)
                W_new[k_prime][k] = logit_s #1.0 * self.cal_w_k_k_prime(k, k_prime, Z)
        #amax = numpy.amax(W_new)
        #amin = numpy.amin(W_new)
        # print W_new
        #W_new = self.convert_range(W_new, (amin-amax), (amax-amin))
        prob_new = numpy.exp(self.log_likelihood_Y(self._Y, self._Z, W_new))

        # compute the probability of using old features
        prob_old = numpy.exp(self.log_likelihood_Y(self._Y, self._Z, W_old))

        # compute the probability of generating new features
        prob_new = prob_new / (prob_old + prob_new)

        # if we accept the proposal, we will replace old W matrix
        if random.random() < prob_new:
            # construct A_new and Z_new
            self._W = numpy.copy(W_new)
            return True

        return False
                #else:
                #    W_new[k][k_prime] = 1.0 * self.cal_w_k_k_prime(k, k_prime, Z)
                #    W_new[k_prime][k] = 1.0 * self.cal_w_k_k_prime(k, k_prime, Z)
        #amax = numpy.amax(W_new)
        #amin = numpy.amin(W_new)
        #print W_new
        #W_new = self.convert_range(W_new, (amin-amax), (amax-amin))
        #W_new = preprocessing.normalize(W_new, norm='l2')
        #W_normed = (W_new - W_new.min(0)) / W_new.ptp(0)
        #print W_new
        #return  W_new

    def cal_w_k_k_prime(self, k, k_prime, Z):
        w_k_k_prime = 0
        w_k_k_prime_0 = 0
        for i in range(self._N):
            for j in range(self._N):
                if (self._Y[i][j] == 1) and (Z[i][k] == 1) and (Z[j][k_prime] == 1):
                    w_k_k_prime += 1
                if (self._Y[i][j] == 0) and (Z[i][k] == 1) and (Z[j][k_prime] == 1):
                    w_k_k_prime_0 += 1
        sample =  numpy.random.beta(w_k_k_prime + 1.0, w_k_k_prime_0 + 1.0, 1)[0]
        logit_s = self.logit(sample)
        return logit_s

    """
    remove the empty column in matrix Z and the corresponding feature in A
    """
    def regularize_matrices(self):
        assert(self._Z.shape == (self._N, self._K))
        Z_sum = numpy.sum(self._Z, axis=0)
        assert(len(Z_sum) == self._K)
        indices = numpy.nonzero(Z_sum == 0)
        #assert(numpy.min(indices)>=0 and numpy.max(indices)<self._K)
        
        #print self._K, indices, [k for k in range(self._K) if k not in indices]

        self._Z = self._Z[:, [k for k in range(self._K) if k not in indices[0]]]
        self._W = self._W[[k for k in range(self._K) if k not in indices[0]], :]

        self._W = self._W[:, [k for k in range(self._K) if k not in indices[0]]]


        self._K = self._Z.shape[1]
        self.sample_W(self._Z)

        assert(self._Z.shape == (self._N, self._K))
        assert(self._W.shape == (self._K, self._K))

    """
    compute the log-likelihood of the data X
    @param X: a 2-D numpy array
    @param Z: a 2-D numpy boolean array
    @param A: a 2-D numpy array, integrate A out if it is set to None
    """
    def log_likelihood_Y(self, Y=None, Z=None, W=None):
        if W is None:
            W = self._W
        if Z is None:
            Z = self._Z
        if Y is None:
            Y = self._Y
            
        assert(Y.shape[0] == Z.shape[0])
        #(N, N) = Y.shape
        (N, K) = Z.shape
        assert(W.shape == (K, K))
        log_likelihood = 1.0

        for i in range(N):
            for j in range(i, N):
                #temp = 0
                #for k in range(K):
                #    for k_prime in range(K):
                #        temp += Z[i][k] * W[k][k_prime] * Z[j][k_prime]
                temp = numpy.dot(numpy.dot(Z[i, :], W),Z[j, :].transpose())
                if Y[i][j] == 1:
                    log_likelihood *= self.sigmoid(temp)
                else:
                    log_likelihood *= (1.0 - self.sigmoid(temp))

        return numpy.log(log_likelihood)
    
    """
    compute the log-likelihood of W
    """
    def log_likelihood_W(self):
        log_likelihood = -0.5 * self._K * self._K * numpy.log(2 * numpy.pi * self._sigma_w * self._sigma_w)
        #for k in range(self._K):
        #    A_prior[k, :] = self._mean_a[0, :]
        #W_prior = numpy.tile(self._W_prior, (self._K, 1))
        log_likelihood -= numpy.trace(numpy.dot((self._W).transpose(), (self._W))) * 0.5 / (self._sigma_w ** 2)
        
        return log_likelihood
    
    """
    compute the log-likelihood of the model
    """
    def log_likelihood_model(self):
        return self.log_likelihood_Y() # + self.log_likelihood_W() + self.log_likelihood_Z()

    """
    sample noise variances, i.e., sigma_y
    """
    def sample_sigma_y(self, sigma_x_hyper_parameter):
        return self.sample_sigma(self._sigma_x_hyper_parameter, self._X - numpy.dot(self._Z, self._A))
    
    """
    sample feature variance, i.e., sigma_a
    """
    def sample_sigma_a(self, sigma_w_hyper_parameter):
        return self.sample_sigma(self._sigma_w_hyper_parameter, self._W - numpy.tile(self._W_prior, (self._K, 1)))

    # Sigmoid function
    def sigmoid(self, x):
        # compute and return the sigmoid activation value for a
        # given input value
        return 1.0 / (1 + numpy.exp(-x))

    def logit(self, p):
        return numpy.log(p / (1-p))

    def convert_range(self, matrix, new_min, new_max):
        old_min = numpy.amin(matrix)
        old_max = numpy.amax(matrix)

        if old_min != old_max:
            for i in range(self._K):
                for j in range(self._K):
                    matrix[i][j] = ((matrix[i][j] - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
        return matrix

    def load_lazega_friend(self):
        f = open('../data/LazegaLawyers/ELfriend36.dat')
        data = numpy.loadtxt(f)
        return data.astype(numpy.int)

    def load_lazega_adv(self):
        f = open('../data/LazegaLawyers/ELadv36.dat')
        data = numpy.loadtxt(f)
        return data.astype(numpy.int)

    def load_lazega_work(self):
        f = open('../data/LazegaLawyers/ELwork36.dat')
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

"""
run IBP on the synthetic 'cambridge bars' dataset, used in the original paper.
"""
if __name__ == '__main__':
    import scipy.io
    #import util.scaled_image
    
    # load the data from the matrix
    #mat_vals = scipy.io.loadmat('../data/cambridge-bars/block_image_set.mat')
    #true_weights = mat_vals['trueWeights']
    #features = mat_vals['features']
    #data = mat_vals['data']
    
    #print true_weights.shape, features.shape, data.shape
    
    # set up the hyper-parameter for sampling alpha
    alpha_hyper_parameter = (1., 1.)
    # set up the hyper-parameter for sampling sigma_x
    sigma_y_hyper_parameter = (1., 1.)
    # set up the hyper-parameter for sampling sigma_a
    sigma_w_hyper_parameter = (1., 1.)
    
    #features = features.astype(numpy.int)
    """
    data = numpy.array([[1, 0, 0, 1, 1],
                        [0, 1, 1, 1, 0],
                        [0, 1, 1, 0, 1],
                        [1, 1, 0, 1, 1],
                        [1, 0, 1, 1, 1]])

    data = numpy.array([[1, 0, 1, 0, 1, 0, 0, 1, 1],
                        [0, 1, 0, 0, 0, 0, 1, 0, 1],
                        [1, 0, 1, 0, 0, 0, 1, 0, 1],
                        [0, 0, 0, 1, 0, 0, 0, 1, 0],
                        [1, 0, 0, 0, 1, 0, 1, 0, 1],
                        [0, 0, 0, 0, 0, 1, 0, 1, 0],
                        [0, 1, 1, 0, 1, 0, 1, 0, 0],
                        [1, 0, 0, 1, 0, 1, 0, 1, 1],
                        [1, 1, 1, 0, 1, 0, 0, 1, 1]])
    """
    # initialize the model
    #ibp = UncollapsedGibbsSampling(10)
    ibp = UncollapsedGibbsSampling(alpha_hyper_parameter, sigma_y_hyper_parameter, sigma_w_hyper_parameter, True)
    #ibp = UncollapsedGibbsSampling(alpha_hyper_parameter)
    data = ibp.load_kinship()
    #data = ibp.load_lazega_friend()
    ibp._initialize(data, 1.0, 1.0, 0.5, None, None, None)
    #ibp._initialize(data[0:1000, :], 1.0, 1.0, 1.0, None, features[0:1000, :])
    #print ibp._Z, "\n", ibp._A
    ibp.sample(1000)
    
    print ibp._Z.sum(axis=0)
    print ibp.log_likelihood_model()
