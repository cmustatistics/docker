from tqdm import tqdm
import tensorflow as tf
from logger import Logger
from rbf import K, K_diag, Ksym
from math import pi
from timeit import default_timer as timer


class IFSGP():
    """
    Information Filter for Sparse Gaussian Processes regression

    The names of the variables follow the notation used in
    the algorithm presented in the supplementary material
    """

    def __init__(self,
                 inducing_points_init,
                 variance_init,
                 lenghscale_init,
                 gaussian_noise_init,
                 float32=False,
                 seed=123):

        tf.compat.v1.reset_default_graph()
        tf.random.set_seed(seed)

        self.dtype = tf.float32 if float32 else tf.float64

        self.pi = tf.constant(pi, dtype=self.dtype)

        # variational parameters
        self._inducing_points = tf.Variable(inducing_points_init,
                                            dtype=self.dtype)  # inducing points
        self.M = inducing_points_init.shape[0]
        self.D = inducing_points_init.shape[1]

        # model parameters

        self._lengthscales_var = self._init_inv_softplus(1 / lenghscale_init)
        self._variance_var = self._init_inv_softplus(variance_init)
        self._gaussian_noise_var = self._init_inv_softplus(gaussian_noise_init)

        # optimization parameters

        self.vars = [self._lengthscales_var,
                     self._variance_var,
                     self._gaussian_noise_var,
                     self._inducing_points]

        # init model parameters
        Krr = Ksym(inv_lengthscales=self._inv_lengthscales(),
                   variance=self._variance(),
                   X=self._inducing_points)
        self.etark, self.Lambdarkr, self.Ck = self._reset(Krr)
        self.Lr = self.Ck

    @tf.function
    def _reset(self, Krr):

        etauk = tf.zeros((self.M, 1), dtype=self.dtype)
        Lambdauku = Krr
        Ck = tf.linalg.cholesky(
            Krr + tf.eye(self.M, dtype=self.dtype) * 1e-6)

        return etauk, Lambdauku, Ck

    def _init_inv_softplus(self, value):
        return tf.Variable(tf.math.log(tf.exp(value) - 1), self.dtype)

    @tf.function
    def _gaussian_noise(self):
        return tf.math.softplus(self._gaussian_noise_var)

    @property
    def gaussian_noise(self):
        return self._gaussian_noise().numpy()

    @property
    def inducing_points(self):
        return self._inducing_points.numpy()

    @tf.function
    def _inv_lengthscales(self):
        return tf.math.softplus(self._lengthscales_var)

    @property
    def lengthscales(self):
        return (1 / self._inv_lengthscales()).numpy()

    @property
    def inv_lengthscales(self):
        return self._inv_lengthscales().numpy()

    @tf.function
    def _variance(self):
        return tf.math.softplus(self._variance_var)

    @property
    def variance(self):
        return self._variance().numpy()

    ########################################################
    # Functions required for inference in all methods
    ########################################################

    @tf.function
    def _infer(self, gaussian_noise,
               Kxr, Krr, trKxx,
               Yk, batch_size,
               prevEtark,
               prevLambdarkr,
               prevCk,
               jitter=1e-6):

        id = tf.eye(self.M, dtype=self.dtype)
        Krx = tf.transpose(Kxr)

        Ak = Krx @ Kxr / gaussian_noise
        bk = Krx @ Yk / gaussian_noise

        Lambdarkr = Ak + prevLambdarkr
        etark = bk + prevEtark

        Gk = tf.linalg.triangular_solve(prevCk, Krx)
        Gkt = tf.transpose(Gk)

        rk = Yk - Gkt @ tf.linalg.triangular_solve(prevCk, prevEtark)

        mse = tf.reduce_sum(tf.square(rk))

        Ok = id + Gk @ Gkt / gaussian_noise
        Ek = tf.linalg.cholesky(Ok)

        Ck = prevCk @ Ek

        pk = tf.linalg.triangular_solve(Ck, Krx @ rk) / gaussian_noise
        gamma = tf.reduce_sum(tf.square(pk))

        logdet = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(Ek)))

        dk = - 0.5 * batch_size * tf.math.log(2 * self.pi * gaussian_noise) \
             - logdet - 0.5 * mse / gaussian_noise + 0.5 * gamma

        Lr = tf.linalg.cholesky(Krr + id * jitter)

        Qx = tf.linalg.triangular_solve(Lr, Krx)

        t = tf.reduce_sum(tf.square(Qx))

        ak = 0.5 * (trKxx - t) / gaussian_noise

        lk = ak - dk  # equiv. to lk = dk - ak if we would maximize

        return etark, Lambdarkr, Ck, lk, mse, ak, dk

    @tf.function
    def _compute_matrices(self, Xk):

        inv_lengthscales = self._inv_lengthscales()
        variance = self._variance()
        inducing_points = self._inducing_points

        Kxr = K(inv_lengthscales=inv_lengthscales, variance=variance,
                X=Xk,
                X2=inducing_points)
        Krr = Ksym(inv_lengthscales=inv_lengthscales, variance=variance,
                   X=inducing_points)
        trKxx = tf.reduce_sum(K_diag(variance=variance, X=Xk))

        return Kxr, Krr, trKxx

    @tf.function
    def _infer_with_reset(self,
                          gaussian_noise,
                          Kxr, Krr, trKxx,
                          Yk, batch_size,
                          jitter=1e-6):
        """
        Inference function optimized for the case when
        prevEtark = 0, prevLambdarkr = Krr and prevCk = Lr
        """

        id = tf.eye(self.M, dtype=self.dtype)
        Krx = tf.transpose(Kxr)

        Lr = tf.linalg.cholesky(Krr + id * jitter)

        Ak = Krx @ Kxr / gaussian_noise
        bk = Krx @ Yk / gaussian_noise

        Lambdarkr = Ak + Krr
        etark = bk

        Gk = tf.linalg.triangular_solve(Lr, Krx)
        Gkt = tf.transpose(Gk)

        rk = Yk

        mse = tf.reduce_sum(tf.square(rk))

        Ok = id + Gk @ Gkt / gaussian_noise
        Ek = tf.linalg.cholesky(Ok)

        Ck = Lr @ Ek

        pk = tf.linalg.triangular_solve(Ck, Krx @ rk) / gaussian_noise
        gamma = tf.reduce_sum(tf.square(pk))

        logdet = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(Ek)))

        dk = - 0.5 * batch_size * tf.math.log(2 * self.pi * gaussian_noise) \
             - logdet - 0.5 * mse / gaussian_noise + 0.5 * gamma

        Qx = tf.linalg.triangular_solve(Lr, Krx)

        t = tf.reduce_sum(tf.square(Qx))

        ak = 0.5 * (trKxx - t) / gaussian_noise

        lk = ak - dk  # since we minimize

        return etark, Lambdarkr, Ck, lk, mse, ak, dk

    @tf.function
    def infer_with_reset(self, Xk, Yk):

        Kxr, Krr, trKxx = self._compute_matrices(Xk=Xk)

        return self._infer_with_reset(
            gaussian_noise=self._gaussian_noise(),
            Kxr=Kxr,
            Krr=Krr,
            trKxx=trKxx,
            Yk=Yk,
            batch_size=Xk.shape[0])

    @tf.function
    def infer_with_prior(self, Xk, Yk, prevEtark, prevLambdarkr, prevCk):

        Kxr, Krr, trKxx = self._compute_matrices(Xk=Xk)

        return self._infer(
            gaussian_noise=self._gaussian_noise(),
            Kxr=Kxr,
            Krr=Krr,
            trKxx=trKxx,
            Yk=Yk,
            batch_size=Xk.shape[0],
            prevEtark=prevEtark,
            prevLambdarkr=prevLambdarkr,
            prevCk=prevCk)

    ########################################################
    # Functions required for prediction
    ########################################################

    def _predict(self, Xs, full_cov=False):
        return self.__predict(Xs=Xs,
                              Ck=self.Ck,
                              Lr=self.Lr,
                              etauk=self.etark,
                              gaussian_noise=self._gaussian_noise(),
                              full_cov=full_cov)

    @tf.function
    def __predict(self, Xs, Ck, Lr, etauk, gaussian_noise, full_cov=False):

        inv_lengthscales = self._inv_lengthscales()
        variance = self._variance()

        Krs = K(inv_lengthscales=inv_lengthscales,
                variance=variance,
                X=self._inducing_points,
                X2=Xs)

        iCkKrs = tf.linalg.triangular_solve(Ck, Krs)

        muS = tf.transpose(iCkKrs) @ tf.linalg.triangular_solve(Ck, etauk)

        iLrKrs = tf.linalg.triangular_solve(Lr, Krs)

        if full_cov:
            Kss = Ksym(inv_lengthscales=inv_lengthscales, variance=variance,
                       X=Xs)
            SigmaS = Kss \
                     - tf.transpose(iLrKrs) @ iLrKrs \
                     + tf.transpose(iCkKrs) @ iCkKrs

            # since we do the Y prediction
            SigmaS += tf.eye(self.M, dtype=self.dtype) * gaussian_noise

        else:
            Kss_diag = K_diag(variance=variance, X=Xs)

            SigmaS = Kss_diag \
                     - tf.reduce_sum(tf.square(iLrKrs), axis=0) \
                     + tf.reduce_sum(tf.square(iCkKrs), axis=0)

            SigmaS = tf.expand_dims(SigmaS, axis=-1)

            # since we do the Y prediction
            SigmaS += gaussian_noise

        return muS, SigmaS

    def predict(self, Xs, full_cov=False):
        mean, var = self._predict(Xs, full_cov)
        return tf.squeeze(mean).numpy(), var.numpy()

    ########################################################
    # Functions required for LIA
    ########################################################

    @tf.function
    def grad_with_reset(self, optimizer, vars, Xk, Yk):

        with tf.GradientTape(watch_accessed_variables=False) as t:
            t.watch(vars)

            etark, Lambdarkr, Ck, lk, mse, ak, dk = \
                self.infer_with_reset(Xk=Xk, Yk=Yk)

        grad = t.gradient(lk, vars)

        optimizer.apply_gradients(zip(grad, vars))

        return etark, Lambdarkr, Ck, lk, mse, ak, dk, grad

    ########################################################
    # Functions required for LIA -> LIF or LIF
    ########################################################

    @tf.function
    def _approximate_grad_with_reset(self, optimizer, vars, Xk, Yk):

        with tf.GradientTape(watch_accessed_variables=False,
                             persistent=True) as t:
            t.watch(vars)

            etark, Lambdarkr, Ck, lk, mse, ak, dk = self.infer_with_reset(Xk=Xk,
                                                                          Yk=Yk)

        grad = t.gradient(lk, vars)
        etark_dvars = t.jacobian(etark, vars, parallel_iterations=20)
        Ck_dvars = t.jacobian(Ck, vars, parallel_iterations=20)

        del t

        optimizer.apply_gradients(zip(grad, vars))

        return etark, Lambdarkr, Ck, lk, mse, ak, dk, grad, etark_dvars, Ck_dvars

    @tf.function
    def _approximate_grad_with_prior(self, optimizer, vars, Xk, Yk, prevEtark,
                                     prevLambdarkr,
                                     prevCk, etark_dprev_lenthscales,
                                     etark_dprev_variance,
                                     etark_dprev_gaussian_noise,
                                     etark_dprev_inducing_points,
                                     Ck_dprev_lenthscales,
                                     Ck_dprev_variance,
                                     Ck_dprev_gaussian_noise,
                                     Ck_dprev_inducing_points
                                     ):

        extended_vars = vars + [prevEtark, prevCk]

        with tf.GradientTape(watch_accessed_variables=False,
                             persistent=True) as t:
            t.watch(extended_vars)

            etark, Lambdarkr, Ck, lk, mse, ak, dk = \
                self.infer_with_prior(Xk=Xk, Yk=Yk,
                                      prevEtark=prevEtark,
                                      prevLambdarkr=prevLambdarkr,
                                      prevCk=prevCk)

        lk_dvars = t.gradient(lk, vars)

        etark_dvars = t.jacobian(etark, vars, parallel_iterations=20)
        Ck_dvars = t.jacobian(Ck, vars, parallel_iterations=20)

        lk_dprev = t.gradient(lk, [prevEtark, prevCk])

        del t

        etark_dprev_vars = [etark_dprev_lenthscales,
                            etark_dprev_variance,
                            etark_dprev_gaussian_noise,
                            etark_dprev_inducing_points]
        Ck_dprev_vars = [Ck_dprev_lenthscales,
                         Ck_dprev_variance,
                         Ck_dprev_gaussian_noise,
                         Ck_dprev_inducing_points]

        for i in range(len(lk_dvars)):
            lk_dvars[i] = lk_dvars[i] \
                          + tf.reshape(
                tf.reduce_sum(tf.transpose(etark_dprev_vars[i])
                              * tf.transpose(lk_dprev[0]), axis=[-1, -2]),
                lk_dvars[i].shape) \
                          + tf.reshape(
                tf.reduce_sum(tf.transpose(Ck_dprev_vars[i])
                              * tf.transpose(lk_dprev[1]), axis=[-1, -2]),
                lk_dvars[i].shape)

        optimizer.apply_gradients(zip(lk_dvars, vars))

        return etark, Lambdarkr, Ck, lk, mse, ak, dk, lk_dvars, etark_dvars, Ck_dvars

    def approximate_grad(self, it, reset, Xk, Yk, optimizer):
        if reset:
            etark, Lambdarkr, Ck, lk, mse, ak, dk, grad, etark_dvars, Ck_dvars = \
                self._approximate_grad_with_reset(optimizer, self.vars,
                                                  Xk, Yk)

        else:
            etark, Lambdarkr, Ck, lk, mse, ak, dk, grad, etark_dvars, Ck_dvars = \
                self._approximate_grad_with_prior(
                    optimizer, self.vars, Xk, Yk,
                    prevEtark=self.etark,
                    prevLambdarkr=self.Lambdarkr,
                    prevCk=self.Ck,
                    etark_dprev_lenthscales=self.etark_dvars[0],
                    etark_dprev_variance=self.etark_dvars[1],
                    etark_dprev_gaussian_noise=self.etark_dvars[2],
                    etark_dprev_inducing_points=self.etark_dvars[3],
                    Ck_dprev_lenthscales=self.Ck_dvars[0],
                    Ck_dprev_variance=self.Ck_dvars[1],
                    Ck_dprev_gaussian_noise=self.Ck_dvars[2],
                    Ck_dprev_inducing_points=self.Ck_dvars[3])

        self.etark_dvars = etark_dvars
        self.Ck_dvars = Ck_dvars

        return etark, Lambdarkr, Ck, lk, mse, ak, dk, grad

    ########################################################
    # Functions required for LIA -> Posterior Propagation
    ########################################################

    @tf.function
    def _posterior_propagation_reset(self, Xk, Yk, jitter=1e-6):

        gaussian_noise = self._gaussian_noise()
        inv_lengthscales = self._inv_lengthscales()
        variance = self._variance()
        inducing_points = self._inducing_points

        Kxr = K(inv_lengthscales=inv_lengthscales, variance=variance,
                X=Xk,
                X2=inducing_points)
        Krr = Ksym(inv_lengthscales=inv_lengthscales, variance=variance,
                   X=inducing_points)

        id = tf.eye(self.M, dtype=self.dtype)
        Krx = tf.transpose(Kxr)

        Lr = tf.linalg.cholesky(Krr + id * jitter)

        Ak = Krx @ Kxr / gaussian_noise
        bk = Krx @ Yk / gaussian_noise

        Lambdarkr = Ak + Krr
        etark = bk

        Gk = tf.linalg.triangular_solve(Lr, Krx)
        Gkt = tf.transpose(Gk)

        Ok = id + Gk @ Gkt / gaussian_noise
        Ek = tf.linalg.cholesky(Ok)

        Ck = Lr @ Ek

        return etark, Lambdarkr, Ck

    @tf.function
    def _posterior_propagation_with_prior(self, Xk, Yk, prevLambdarkr,
                                          prevEtark, prevCk):

        gaussian_noise = self._gaussian_noise()
        inv_lengthscales = self._inv_lengthscales()
        variance = self._variance()
        inducing_points = self._inducing_points

        Kxr = K(inv_lengthscales=inv_lengthscales, variance=variance,
                X=Xk,
                X2=inducing_points)

        id = tf.eye(self.M, dtype=self.dtype)
        Krx = tf.transpose(Kxr)

        Ak = Krx @ Kxr / gaussian_noise
        bk = Krx @ Yk / gaussian_noise

        Lambdarkr = Ak + prevLambdarkr
        etark = bk + prevEtark

        Gk = tf.linalg.triangular_solve(prevCk, Krx)
        Gkt = tf.transpose(Gk)

        Ok = id + Gk @ Gkt / gaussian_noise
        Ek = tf.linalg.cholesky(Ok)

        Ck = prevCk @ Ek

        return etark, Lambdarkr, Ck

    def posterior_propagation(self, it, reset, Xk, Yk, optimizer):

        if reset:
            etark, Lambdarkr, Ck = self._posterior_propagation_reset(Xk=Xk,
                                                                     Yk=Yk)
        else:
            etark, Lambdarkr, Ck = self._posterior_propagation_with_prior(
                Xk=Xk, Yk=Yk,
                prevEtark=self.etark,
                prevLambdarkr=self.Lambdarkr,
                prevCk=self.Ck)

        return etark, Lambdarkr, Ck, None, None, None, None, None

    ########################################################
    # Optimization
    ########################################################

    def optimize(self,
                 iterations,
                 Xtrain, Ytrain, Xval, Yval,
                 path,
                 run=0,
                 batch_size=None,
                 collect_every=1,
                 use_prior_threshold=False,
                 reset_prior_every_it=False,
                 use_prev_grad=False,
                 switch_to_prev_grad=False,
                 switch_at_epoch=5,
                 lr=0.001,
                 Ystd=None):

        Xtrain = tf.convert_to_tensor(Xtrain, dtype=self.dtype)
        Ytrain = tf.convert_to_tensor(Ytrain, dtype=self.dtype)
        Xval = tf.convert_to_tensor(Xval, dtype=self.dtype)
        Yval = tf.convert_to_tensor(Yval, dtype=self.dtype)

        N = Xtrain.shape[0]
        if batch_size is None:
            batch_size = N

        ########################
        # Configure IFSGP in oder to obtain LIA, LIA -> LIF, LIF or LIA -> PP
        #######################
        name = None
        update_with_grad = None

        if reset_prior_every_it:
            name = 'LIA'

            update_with_grad = lambda it, reset, Xk, Yk, optimizer: \
                self.grad_with_reset(optimizer=optimizer, vars=self.vars, Xk=Xk,
                                     Yk=Yk)

            if use_prior_threshold:
                if switch_to_prev_grad:
                    name = 'LIA_to_LIF'
                else:
                    name = 'LIA_to_PP'
        elif use_prev_grad:
            name = 'LIF'

            update_with_grad = self.approximate_grad

        assert name is not None
        assert update_with_grad is not None

        thresholded_update = update_with_grad

        ########################################################
        # Initialize optimizer and logger
        ########################################################

        optimizer = tf.optimizers.Adam(
            learning_rate=lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7)

        log_dir = '{0}{1}_{2}_{3}_{4}/run_{5}'.format(
            path, name, self.M, batch_size, lr, run)
        logger = Logger(log_dir=log_dir, model=self,
                        Xtrain=Xtrain, Ytrain=Ytrain,
                        Xval=Xval, Yval=Yval,
                        Ystd=Ystd)
        logger.collect(it=0, data_points_used=0)

        ########################################################
        # Set up training dataset
        ########################################################

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (Xtrain, Ytrain)).shuffle(10000).repeat()
        train_iter = iter(train_dataset.batch(batch_size))

        reset_every = tf.math.ceil(N / batch_size)  # iterations per epoch

        iterations_threshold = None
        if use_prior_threshold:
            ########################################################
            # We compute the number of iterations at which we should
            # switch from LIA to the posterior propagation in
            # order to run the posterior propagation for switch_at_epoch epochs
            ########################################################

            iterations_threshold = (tf.math.floor(
                iterations / reset_every) - switch_at_epoch) * reset_every
            print(
                "The posterior propagation will start at iteration {0}".format(
                    iterations_threshold + 1))
            print("Thus, it will run for {0} epochs".format(
                (iterations - iterations_threshold) / reset_every))

        should_reset = lambda it: (it % reset_every == 0)
        should_record = lambda it: (it == 0) \
                                   or ((it + 1) % collect_every == 0) \
                                   or (it == iterations - 1)

        for it in tqdm(range(iterations)):

            Xk, Yk = next(train_iter)

            reset = should_reset(it)

            if use_prior_threshold and reset and it >= iterations_threshold:

                ########################################################
                # We switch to optimize LIF, in which case we reset ADAM,
                # or switch to do the posterior propagation (PP) depending
                # on the configuration parameters
                ########################################################

                if switch_to_prev_grad:

                    # Reset optimizer variables
                    for var in optimizer.variables():
                        var.assign(tf.zeros_like(var))

                    thresholded_update = self.approximate_grad

                else:

                    thresholded_update = self.posterior_propagation

                use_prior_threshold = False
                should_reset = lambda it: False

                ########################################################
                # We increase the recording frequency in order to be
                # able to note the effect of switching
                ########################################################

                sample = tf.math.ceil(reset_every / 5)
                should_record = lambda it: (it == iterations_threshold) or (
                            it % sample == 0) or (it == iterations - 1)

                print('Changed to posterior propagation at iteration {0}'.format(it + 1))


            ########################################################
            # Time the method
            ########################################################

            start = timer()

            etark, Lambdarkr, Ck, lk, mse, ak, dk, grad = \
                thresholded_update(it=it, reset=reset, Xk=Xk, Yk=Yk,
                                   optimizer=optimizer)

            runtime = timer() - start

            self.etark = etark
            self.Lambdarkr = Lambdarkr
            self.Ck = Ck

            ########################################################
            # Collect stats
            ########################################################
            record = should_record(it)
            if record:
                ########################################################
                # The following computations are only necessary
                # due to the logging while training
                ########################################################
                Krr = Ksym(inv_lengthscales=self._inv_lengthscales(),
                           variance=self._variance(),
                           X=self._inducing_points)
                self.Lr = tf.linalg.cholesky(
                    Krr + tf.eye(self.M, dtype=self.dtype) * 1e-6)

            logger.collect(lk=lk, ak=ak, dk=dk,
                           grad=grad,
                           it=it + 1,
                           record=record,
                           data_points_used=batch_size*(it+1),
                           number_of_batches=1,
                           runtime=runtime)
