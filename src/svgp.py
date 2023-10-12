from logger import Logger
from tqdm import tqdm
from math import pi
import tensorflow as tf
import gpflow
from timeit import default_timer as timer


class SVGP():

    def __init__(self,
                 inducing_points_init,
                 variance_init,
                 lenghscale_init,
                 gaussian_noise_init,
                 float32=False,
                 seed=123):

        tf.compat.v1.reset_default_graph()
        tf.random.set_seed(seed)

        if float32:
            self.dtype = tf.float32
        else:
            self.dtype = tf.float64

        gpflow.config.set_default_float(self.dtype)

        self.D = inducing_points_init.shape[1]
        self.M = inducing_points_init.shape[0]
        self.pi = tf.cast(pi,dtype=self.dtype)

        self.model = gpflow.models.SVGP(
            whiten=False,
            kernel=gpflow.kernels.RBF(
                lengthscales=tf.cast(lenghscale_init,dtype=self.dtype),
                variance=tf.cast(variance_init,dtype=self.dtype)
            ),
            likelihood=gpflow.likelihoods.Gaussian(variance=gaussian_noise_init),
            inducing_variable=inducing_points_init
        )

    @property
    def lengthscales(self):
        return self.model.kernel.lengthscales.numpy()

    @property
    def inv_lengthscales(self):
        return (1/self.model.kernel.lengthscales).numpy()

    @property
    def variance(self):
        return self.model.kernel.variance.numpy()

    @property
    def _inducing_points(self):
        return self.model.inducing_variable.Z

    @property
    def inducing_points(self):
        return self.model.inducing_variable.Z.numpy()

    @property
    def gaussian_noise(self):
        return self.model.likelihood.variance.numpy()

    def _predict(self,Xs,full_cov=False):
        mean,covar = self.model.predict_y(Xnew=Xs,full_cov=full_cov)

        return mean,covar

    def predict(self,X,full_cov=False):
       mean,covar = self._predict(X,full_cov)
       return mean.numpy(),covar.numpy()

    def optimize(self,
                 iterations,
                 Xtrain, Ytrain, Xval, Yval,
                 path,
                 run=0,
                 batch_size=None,
                 collect_every=1,
                 lr = 0.001,
                 gamma=0.1,
                 nat_grad=False,
                 Ystd=None):

        Xtrain = tf.convert_to_tensor(Xtrain, dtype=self.dtype)
        Ytrain = tf.convert_to_tensor(Ytrain, dtype=self.dtype)
        Xval = tf.convert_to_tensor(Xval, dtype=self.dtype)
        Yval = tf.convert_to_tensor(Yval, dtype=self.dtype)

        N = Xtrain.shape[0]
        if batch_size is None:
            batch_size = N

        ########################
        # Set up logger
        #######################
        log_dir = '{0}SVI_{1}_{2}_{3}{4}/run_{5}'.format(
            path,
            self.M,
            batch_size,
            lr,
            '_nat-grad_{0}'.format(gamma) if nat_grad else '',
            run)
        logger = Logger(log_dir=log_dir, model=self,
                        Xtrain=Xtrain, Ytrain=Ytrain, Xval=Xval, Yval=Yval,
                        Ystd=Ystd)
        #######################

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (Xtrain, Ytrain)).shuffle(10000).repeat()
        train_iter = iter(train_dataset.batch(batch_size))
        training_loss = self.model.training_loss_closure(train_iter,compile=True)

        optimizer = tf.optimizers.Adam(
            learning_rate=lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7)

        nat_optimizer, variational_variables = None, None
        if nat_grad:
            gpflow.utilities.set_trainable(self.model.q_mu,False)
            gpflow.utilities.set_trainable(self.model.q_sqrt,False)
            variational_variables = [(self.model.q_mu,self.model.q_sqrt)]
            nat_optimizer = gpflow.optimizers.NaturalGradient(gamma=gamma)

        trainable_variables = self.model.trainable_variables

        logger.collect(it=0,data_points_used=0)

        for it in tqdm(range(iterations)):

            start = timer()

            optimizer.minimize(training_loss,trainable_variables)
            if nat_grad:
                nat_optimizer.minimize(training_loss,variational_variables)

            runtime = timer() - start

            logger.collect(lk=training_loss(),
                           it=it+1,
                           record=(it == 0) or ((it + 1) % collect_every == 0),
                           data_points_used=(it+1)*batch_size,
                           runtime=runtime,
                           number_of_batches=1)