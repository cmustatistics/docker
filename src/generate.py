import GPy
import numpy as np
from dotdic import DotDic
from loader import save_to

def genData(Ntrain, Ntest, Mbuild, D, lengthscale, sig2_noise, sig2_0,lower,upper, seed=123):
    np.random.seed(seed)

    k = GPy.kern.RBF(input_dim=D,lengthscale=lengthscale)
    k.variance = sig2_0

    range_ = upper - lower

    RRm = np.random.rand(Mbuild,D) * range_ + lower
    Xtrain = np.random.rand(Ntrain,D) * range_ + lower
    Xtest = np.random.rand(Ntest,D) * range_ + lower

    mu = np.zeros((Mbuild)) # vector of the means
    C = k.K(RRm,RRm) # covariance matrix
    w = np.random.multivariate_normal(mu,C,1).transpose()

    iKrr = np.linalg.inv(C + np.eye(Mbuild)*1e-7)
    iKrr_w = np.dot(iKrr, w)

    Kxtr = k.K(Xtrain, RRm)
    Kxpr = k.K(Xtest, RRm)

    Ftrain = np.dot(Kxtr, iKrr_w)
    Ftest = np.dot(Kxpr, iKrr_w)

    Y_train = Ftrain + np.random.randn(Ntrain,1) * np.sqrt(sig2_noise)
    Y_test = Ftest + np.random.randn(Ntest,1) * np.sqrt(sig2_noise)

    return Xtrain, Xtest, Y_train, Y_test, Ftrain, Ftest

def save(D,lengthscale,variance,gaussian_noise,
         Ntrain=100000,Ntest=10000,M_build=2000,
         lower = -1,upper = 1,seed = 123):


    Xtrain, Xtest, Ytrain, Ytest, Ftrain, Ftest = \
        genData(Ntrain=Ntrain,
                Ntest=Ntest,
                Mbuild=M_build,
                D=D,
                lengthscale=lengthscale,
                sig2_0=variance,
                sig2_noise=gaussian_noise,
                lower=lower,
                upper=upper,
                seed = seed)

    data = {}
    data['Xtrain'] = Xtrain
    data['Xtest'] = Xtest
    data['Ytrain'] = Ytrain
    data['Ytest'] = Ytest
    data['Ftrain'] = Ftrain
    data['Ftest'] = Ftest

    data['lengthscale'] = np.repeat(lengthscale, D)
    data['variance'] = variance
    data['gaussian_noise'] = gaussian_noise

    data['lower'] = lower
    data['upper'] = upper
    data['D'] = D

    data = DotDic(data)

    name = "gendata_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}".format(
        D,
        Ntrain,
        Ntest,
        lengthscale,
        variance,
        gaussian_noise,
        lower,
        upper)

    print("Saving {0}".format(name))
    save_to(obj=data,
            object_str='data',
            path='./data/{0}/'.format(name))




