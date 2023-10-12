import numpy as np
from loader import load

from ifsgp import IFSGP
from svgp import SVGP
from gp import GP
from vfe import VFE

def compare(data_name,
            M,
            batch_size,
            iterations,
            seed=123,

            run_GP=False,
            run_VFE=False,

            run_IFSGP=False,
            SIFGP_switch_to_prev_grad=False,
            SIFGP_reset_prior_every_it=False,
            SIFGP_use_prev_grad=False,
            SIFGP_use_prior_threshold=False,

            run_SVGP=False,
            SVGP_nat_grad=False,

            collect_every=1,
            use_float32=False,

            lenghscale_init = 1,
            variance_init = 1,
            gaussian_noise_init = 1,
            lr = 0.0001):

    np.random.seed(seed)

    # precision to be used by all methods
    dtype = np.float32 if use_float32 else np.float64

    # path for saving the results
    path = './results/{0}/'.format(data_name)

    ############################
    # Load data
    ############################

    data = load(object_str='data',path='./data/{0}/'.format(data_name))
    data.Xtrain = data.Xtrain.astype(dtype)
    data.Xtest = data.Xtest.astype(dtype)
    data.Ytrain = data.Ytrain.astype(dtype)
    data.Ytest = data.Ytest.astype(dtype)
    data.Ystd = dtype(data.Ystd) if 'Ystd' in data else None

    ############################
    # Print dataset parameters
    ############################

    print('Dataset {0} stats'.format(data_name))

    if 'lengthscale' in data:
        data.lengthscale = dtype(data.lengthscale)
        data.variance = dtype(data.variance)
        data.gaussian_noise = dtype(data.gaussian_noise)

        print('lengthscale {0}'.format(data.lengthscale))
        print('variance {0}'.format(data.variance))
        print('gaussian noise {0}'.format(data.gaussian_noise))


    print('Ntrain {0}'.format(data.Xtrain.shape[0]))
    print('Ntest {0}'.format(data.Xtest.shape[0]))
    print('Dimension {0}'.format(data.D))

    ############################
    # Init Params
    ############################

    lenghscale_init = np.repeat(lenghscale_init, data.D).astype(dtype)
    variance_init = dtype(variance_init)
    gaussian_noise_init = dtype(gaussian_noise_init)

    N = data.Xtrain.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    idx = idx[0:M]
    inducing_points_init = data.Xtrain[idx, ...]
    inducing_points_init = inducing_points_init.astype(dtype)

    ############################
    # Methods
    ############################

    if run_GP:

            gp = GP( variance_init=variance_init,
                     lenghscale_init=lenghscale_init,
                     float32=use_float32,
                     seed=seed,
                     gaussian_noise_init=gaussian_noise_init)

            gp.optimize( iterations=iterations,
                 Xtrain=data.Xtrain,Ytrain=data.Ytrain,Xval=data.Xtest,Yval=data.Ytest,
                 path=path,
                 Ystd=data.Ystd)

    if run_VFE:

            vfe = VFE(variance_init=variance_init,
                    lenghscale_init=lenghscale_init,
                    float32=use_float32,
                    seed=seed,
                    gaussian_noise_init=gaussian_noise_init,
                    inducing_points_init=inducing_points_init)

            vfe.optimize(iterations=iterations,
                        Xtrain=data.Xtrain, Ytrain=data.Ytrain,
                        Xval=data.Xtest, Yval=data.Ytest,
                        path=path,
                        Ystd=data.Ystd)

    ############################
    # SGP Stochastic gradient
    ############################

    if run_IFSGP:

        ifsgp = IFSGP(
            inducing_points_init=inducing_points_init,
            variance_init=variance_init,
            lenghscale_init=lenghscale_init,
            gaussian_noise_init=gaussian_noise_init,
            seed=seed,
            float32=use_float32)

        ifsgp.optimize(
            iterations=iterations,
            Xtrain=data.Xtrain,Ytrain=data.Ytrain,Xval=data.Xtest,Yval=data.Ytest,
            collect_every=collect_every,
            batch_size=batch_size,
            path=path,
            Ystd=data.Ystd,

            lr=lr,
            switch_to_prev_grad=SIFGP_switch_to_prev_grad,
            use_prior_threshold=SIFGP_use_prior_threshold,
            reset_prior_every_it=SIFGP_reset_prior_every_it,
            use_prev_grad=SIFGP_use_prev_grad)

    ############################
    # GPflow SVGP
    ############################
    if run_SVGP:

        svgp = SVGP(
            inducing_points_init=inducing_points_init,
            variance_init=variance_init,
            lenghscale_init=lenghscale_init,
            gaussian_noise_init=gaussian_noise_init,
            seed=seed,
            float32=use_float32)

        svgp.optimize(
            lr=lr,
            iterations=iterations,
            Xtrain=data.Xtrain,Ytrain=data.Ytrain,
            Xval=data.Xtest,Yval=data.Ytest,
            collect_every=collect_every,
            batch_size=batch_size,
            path=path,
            nat_grad = SVGP_nat_grad,
            Ystd=data.Ystd)
