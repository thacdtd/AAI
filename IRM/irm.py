# from BNP.crp import crp
import BNP.crp as crp
import BNP.bb as bb


if __name__ == "__main__":
    print crp.crp(num_customers=20, alpha=2)
    print bb.beta_binormal(1, 1, 10, 4)
