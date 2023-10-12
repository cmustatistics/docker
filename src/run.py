#########################################
# Utils for running experiments
#########################################

from generate import save
from uci import save_data
from dotdic import DotDic
from comparison import compare

def create_toy_datasets(seed=123):
    save(Ntrain=5000, Ntest=500, D=1, lengthscale=0.1, variance=1,
         gaussian_noise=0.01, lower=-1, upper=1, seed=seed)
    save(Ntrain=5000, Ntest=500, D=2, lengthscale=0.2, variance=1,
         gaussian_noise=0.01, lower=-1, upper=1, seed=seed)


def create_synthetic_datasets(seed=123):
    save(Ntrain=100000, Ntest=10000, D=5, lengthscale=0.5, variance=1,
         gaussian_noise=0.01, lower=-1, upper=1, seed=seed)
    save(Ntrain=100000, Ntest=10000, D=10, lengthscale=1, variance=1,
         gaussian_noise=0.01, lower=-1, upper=1, seed=seed)


def create_real_datasets(seed=123):
    save_data("airlines", seed)
    save_data("protein", seed)
    save_data("bike", seed)
    save_data("song", seed)
    save_data("sgemm", seed)
    save_data("gas", seed)
    save_data("buzz", seed)


def defaults(name,
             number_inducing_points=500,
             lengthscale_init=1,
             variance_init=1,
             iterations=40000,
             batch_size=5000,
             learning_rate=0.001,
             collect_every=1000
             ):
    """
    Set the parameters for the experiment
    """
    return DotDic({
        'data_name': name,
        'number_inducing_points': number_inducing_points,
        'lengthscale_init': lengthscale_init,
        'variance_init': variance_init,
        'iterations': iterations,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'collect_every': collect_every
    })


def all_experiments(seed):

    return [

        # Toy experiments

        defaults(name='gendata_1_5000_500_0.1_1_0.01_-1_1',
                 number_inducing_points=20,
                 iterations=20000,
                 variance_init=2,
                 learning_rate=0.001,
                 batch_size=500),

        defaults(name='gendata_2_5000_500_0.2_1_0.01_-1_1',
                 number_inducing_points=20,
                 iterations=20000,
                 variance_init=2,
                 learning_rate=0.001,
                 batch_size=500),

        # Synthetic experiments

        defaults(name='gendata_5_100000_10000_0.5_1_0.01_-1_1',
                 number_inducing_points=100,
                 iterations=10000,
                 variance_init=2),

        defaults(name='gendata_10_100000_10000_1_1_0.01_-1_1',
                 variance_init=2,
                 lengthscale_init=2,
                 iterations=10000),

        # Real data experiments

        defaults(name='sgemm', iterations=15000),
        defaults(name='buzz', iterations=60000, learning_rate=0.0001),
        defaults(name='airlines', iterations=60000, learning_rate=0.0001),
        defaults(name='bike', iterations=40000, learning_rate=0.0001),
        defaults(name='gas', iterations=40000, learning_rate=0.0001),
        defaults(name='protein', iterations=5000),
        defaults(name='song', iterations=60000, learning_rate=0.0001),

    ]


def demo_experiments(dimensions,number_of_inducing_points,seed):

    return [
        defaults(name='gendata_{0}_100000_10000_0.5_1_0.01_-1_1'.format(dimensions),
                 number_inducing_points=number_of_inducing_points,
                 iterations=200,
                 variance_init=2)
    ]


def run(experiment,
        seed=123,

        run_SIFGP=False,
        SIFGP_reset_prior_every_it=False,
        SIFGP_use_prev_grad=False,
        SIFGP_use_prior_threshold=False,
        SIFGP_switch_to_prev_grad=False,

        run_SVI=False,
        SVI_nat_grad=False,

        run_GP=False,
        run_VFE=False):
    compare(data_name=experiment.data_name,
            M=experiment.number_inducing_points,
            batch_size=experiment.batch_size,
            iterations=experiment.iterations,
            seed=seed,
            collect_every=experiment.collect_every,
            lenghscale_init=experiment.lengthscale_init,
            variance_init=experiment.variance_init,
            lr=experiment.learning_rate,

            run_IFSGP=run_SIFGP,
            SIFGP_reset_prior_every_it=SIFGP_reset_prior_every_it,
            SIFGP_use_prev_grad=SIFGP_use_prev_grad,
            SIFGP_use_prior_threshold=SIFGP_use_prior_threshold,
            SIFGP_switch_to_prev_grad=SIFGP_switch_to_prev_grad,

            run_SVGP=run_SVI,
            SVGP_nat_grad=SVI_nat_grad,

            run_GP=run_GP,
            run_VFE=run_VFE)


def GP(experiment, seed):
    run(experiment=experiment,
        seed=seed,
        run_GP=True)


def VFE(experiment, seed):
    run(experiment=experiment,
        seed=seed,
        run_VFE=True)


def LIA(experiment, seed):
    run(experiment=experiment,
        seed=seed,
        run_SIFGP=True,
        SIFGP_reset_prior_every_it=True)


def LIA_TO_LIF(experiment, seed):
    run(experiment=experiment,
        seed=seed,
        run_SIFGP=True,
        SIFGP_reset_prior_every_it=True,
        SIFGP_use_prior_threshold=True,
        SIFGP_switch_to_prev_grad=True)


def LIA_TO_PP(experiment, seed):
    run(experiment=experiment,
        seed=seed,
        run_SIFGP=True,
        SIFGP_reset_prior_every_it=True,
        SIFGP_use_prior_threshold=True)


def LIF(experiment, seed):
    run(experiment=experiment,
        seed=seed,
        run_SIFGP=True,
        SIFGP_use_prev_grad=True)


def SVGP(experiment, seed):
    run(experiment=experiment,
        seed=seed,
        run_SVI=True,
        SVI_nat_grad=True)

