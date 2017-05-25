########### PARAMETERS ################
# alpha: scalar, the alpha parameter of the prior beta distribution
# beta: scalar, the beta parameter of the prior beta distribution
# n: scalar, the number of binomial trials
# k: array/vector-like, the number of successes
# NOTE: this implementation is still limited in some cases where combinations of alpha, beta, n, and k make the gamma function evalute to
#       very large numbers
#######################################
import scipy as sp
from scipy import special


def bb(alpha, beta, n, k):
    part_1 = sp.special.comb(n, k)
    part_2 = sp.special.beta(k + alpha, n - k + beta)
    part_3 = sp.special.beta(alpha, beta)

    return part_1 * (part_2 / part_3)


def beta_binormal(alpha, beta, n, k):
    part_1 = (sp.special.gamma(n+1))/((sp.special.gamma(k+1))*(sp.special.gamma(n+1-k)))
    part_2 = ((sp.special.gamma(k+alpha))*(sp.special.gamma(n-k+beta)))/(sp.special.gamma(n+alpha+beta))
    part_3 = (sp.special.gamma(alpha+beta))/((sp.special.gamma(alpha))*(sp.special.gamma(beta)))
    return part_1*part_2*part_3
