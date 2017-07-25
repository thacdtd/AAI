import numpy as np
from scipy import special

class CPF(object):
    def __init__(self, y, K=10, a=0.3, a_prime=0.3, c=0.3, c_prime=0.3, b_prime=1.0, d_prime=1.0):
        self.y = y
        self.K = K
        self.dim_U = self.y.shape[0]
        self.dim_I = self.y.shape[1]

        self.a = a
        self.a_prime = a_prime
        self.c = c
        self.c_prime = c_prime
        self.b_prime = b_prime
        self.d_prime = d_prime

        self.kappa_shp = self.init_shape(a_prime, a, self.dim_U)
        self.kappa_rte = np.ones((self.dim_U))
        self.tau_shp = self.init_shape(c_prime, c, self.dim_I)
        self.tau_rte = np.ones((self.dim_I))
        self.phi = self.init_phi()
        self.gamma_shp = np.ones((self.dim_U, K))
        self.gamma_rte = np.ones((self.dim_U, K))
        self.lambda_shp = np.ones((self.dim_I, K))
        self.lambda_rte = np.ones((self.dim_I, K))

    def init_shape(self, value1, value2, dim):
        return np.full((dim),value1 + value2*self.K)

    def init_phi(self):
        return np.ones((self.dim_U, self.dim_I, self.K))

    def sum_phi_k(self, u, i):
        sum_phi_k = 0
        for k in range(0, self.K):
            sum_phi_k += self.phi[u][i][k]
        return sum_phi_k

    def update_phi_k(self, gamma_shp_k, gamma_rte_k, lambda_shp_k, lambda_rte_k):
        return np.exp(special.psi(gamma_shp_k) - np.log(gamma_rte_k) + special.psi(lambda_shp_k) - np.log(lambda_rte_k))

    def update_phi(self):
        for u in range(0, self.dim_U):
            for i in range(0, self.dim_I):
                for k in range(0, self.K):
                    self.phi[u][i][k] = self.update_phi_k(self.gamma_shp[u][k], self.gamma_rte[u][k],
                                        self.lambda_shp[i][k], self.lambda_rte[i][k]) / self.sum_phi_k(u, i)
                    #self.phi[u][i][k] = np.random.exponential(temp)

    def update_gamma_u_k(self, u, k):
        shp = self.a
        rte = self.kappa_shp[u]/self.kappa_rte[u]
        for i in range(0, self.dim_I):
            shp += self.y[u][i]*self.phi[u][i][k]
            rte += self.lambda_shp[i][k]/self.lambda_rte[i][k]
        return shp, rte

    def update_kappa_rte_u(self, u):
        kappa_rte_u = self.a_prime/self.b_prime
        for k in range(0, self.K):
            kappa_rte_u += self.gamma_shp[u][k]/self.gamma_rte[u][k]
        return kappa_rte_u

    def update_lambda_i_k(self, i, k):
        shp = self.a
        rte = self.tau_shp[i] / self.tau_rte[i]
        for u in range(0, self.dim_U):
            shp += self.y[u][i] * self.phi[u][i][k]
            rte += self.gamma_shp[u][k]/self.gamma_rte[u][k]
        return shp, rte

    def update_tau_rte_i(self, i):
        tau_rte_i = self.c_prime/self.d_prime
        for k in range(0, self.K):
            tau_rte_i += self.lambda_shp[i][k]/self.lambda_rte[i][k]
        return tau_rte_i

    def update_user(self):
        for u in range(0, self.dim_U):
            for k in range(0,self.K):
                self.gamma_shp[u][k], self.gamma_rte[u][k] = self.update_gamma_u_k(u, k)
            self.kappa_rte[u] = self.update_kappa_rte_u(u)

    def update_item(self):
        for i in range(0, self.dim_I):
            for k in range(0,self.K):
                self.lambda_shp[i][k], self.lambda_rte[i][k] = self.update_lambda_i_k(i, k)
            self.tau_rte[i] = self.update_tau_rte_i(i)


    def fit(self, max_iter=100):
        for iter in range(0, max_iter):
            # update multinominal
            self.update_phi()

            # user
            self.update_user()

            # item
            self.update_item()

###############################################################################

if __name__ == "__main__":
    R = [
         [5,3,0,1],
         [4,0,0,1],
         [1,1,0,5],
         [1,0,0,4],
         [1,0,5,1],
         [1,2,1,5]
        ]

    R = np.array(R)
    print R[0][1]

    N = len(R)
    M = len(R[0])
    K = 2

    cpf = CPF(R,5)

    cpf.fit()

    print cpf.gamma_shp
    print cpf.gamma_rte

    print cpf.lambda_shp
    print cpf.lambda_rte

    user = np.zeros((cpf.dim_U, cpf.K))
    item = np.zeros((cpf.dim_I, cpf.K))
    for k in range(0, cpf.K):
        for u in range(0, cpf.dim_U):
            user[u][k] = np.random.gamma(cpf.gamma_shp[u][k], 1/cpf.gamma_rte[u][k])
        for i in range(0, cpf.dim_I):
            item[i][k] = np.random.gamma(cpf.lambda_shp[i][k], 1/cpf.lambda_rte[i][k])

    predict = np.random.poisson(np.dot(user, item.T))
    print "============================"
    print predict
