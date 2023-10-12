from pathlib import Path
import shutil
import tensorflow as tf

import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import pi
from numpy import array

class Logger():

    def __init__(self, log_dir, model, Xtrain, Ytrain, Xval, Yval,
                 dtype=tf.float64, Ystd=None):

        # Create directory for logs
        shutil.rmtree(log_dir,ignore_errors=True)
        Path(log_dir).mkdir(parents=True)

        self.model = model
        self.dtype = dtype

        self.writer = tf.summary.create_file_writer(log_dir)

        self.pi = tf.constant(pi, dtype=self.dtype)

        # for tracking stats
        self.Xtrain=Xtrain
        self.Ytrain=Ytrain
        self.iter = None
        if Xtrain.shape[0] > 100000:
            batch_size = 100000
            dataset = tf.data.Dataset.from_tensor_slices(
                (Xtrain, Ytrain)).shuffle(100000).repeat()
            self.iter = iter(dataset.batch(batch_size))
        self.Xval = Xval
        self.Yval = Yval

        self.Ystd = tf.cast(Ystd, dtype=self.dtype) \
            if Ystd is not None else None
        self.Yvar = tf.square(self.Ystd) if self.Ystd is not None else None

        if self.model.D == 1:
            self.inducing_points_hist = []
            self.inducing_points_hist_it = []

        self.elapsed_time = tf.cast(0,dtype=tf.float64)

    def next_train_batch(self):
        if self.iter is None:
            return self.Xtrain,self.Ytrain
        else:
            return next(self.iter)

    @tf.function
    def __stats(self,mean,var,Y):

        # mse and avg ll per point
        mse_vector = tf.square(Y - mean)
        mse = tf.reduce_mean(mse_vector)
        ll_val = - 0.5 * tf.math.log(2 * self.pi) \
                 - 0.5 * tf.reduce_mean(mse_vector / var + tf.math.log(var))

        rmse = tf.sqrt(mse)

        # avg coverage per point
        delta = 1.96 * tf.sqrt(var)
        coverage = tf.reduce_mean(
            tf.cast(tf.math.logical_and(Y <= mean + delta, Y >= mean - delta),
                    tf.float32))

        # reductions
        min_var = tf.reduce_min(var)
        max_var = tf.reduce_max(var)
        avg_var = tf.reduce_mean(var)

        return mse,rmse,mse_vector,ll_val,coverage,min_var,max_var,avg_var

    @tf.function
    def __stats_write(self,mse,mse_vector,rmse,ll_val,coverage,min_var,max_var,avg_var,var,step,str,suffix):

        with self.writer.as_default():
            tf.summary.scalar("{0} mse{1}".format(str,suffix), mse, step=step)
            tf.summary.histogram("{0} mse distribution{1}".format(str,suffix), mse_vector,
                                 step=step)
            tf.summary.scalar("{0} rmse{1}".format(str,suffix), rmse, step=step)

            tf.summary.scalar("{0} avg log-likelihood per point{1}".format(str,suffix), ll_val, step=step)
            tf.summary.scalar("{0} coverage{1}".format(str,suffix),coverage,step=step)

            tf.summary.scalar("{0} min var{1}".format(str,suffix), min_var,step=step)
            tf.summary.scalar("{0} max var{1}".format(str,suffix), max_var,step=step)
            tf.summary.scalar("{0} avg var{1}".format(str,suffix), avg_var,step=step)
            tf.summary.histogram("{0} var distribution{1}".format(str,suffix), var,step=step)

            if self.Ystd is not None:
                mse_original_scale = mse * self.Yvar
                rmse_original_scale = rmse * self.Ystd

                tf.summary.scalar("{0} mse original scale{1}".format(str,suffix),
                                  mse_original_scale,
                                  step=step)
                tf.summary.scalar("{0} rmse original scale{1}".format(str,suffix),
                                  rmse_original_scale,
                                  step=step)

    def _stats(self, mean, var, Y, str, it,data_points_used):

        assert mean.shape == Y.shape
        assert var.shape == Y.shape

        mse, rmse, mse_vector, ll_val, coverage, min_var, max_var, avg_var =\
            self.__stats(mean,var,Y)

        self.__stats_write(mse,mse_vector,rmse,ll_val,coverage,min_var,max_var,avg_var,var,
                           it,str=str,suffix="")


    def stats(self,it,data_points_used):

        train_label = 'train'
        val_label = 'val'

        Xtrain,Ytrain = self.next_train_batch()

        mean_train, var_train = self.model._predict(Xs=Xtrain)
        self._stats(mean_train,var_train,Ytrain,train_label,it,
                    data_points_used=data_points_used)

        mean_val, var_val = self.model._predict(Xs=self.Xval)
        self._stats(mean_val, var_val, self.Yval,val_label, it,
                    data_points_used=data_points_used)

        # if self.model.D == 1:
        #     self.plot(self.fit(X=Xtrain,Y=Ytrain,mean=mean_train,var=var_train),
        #           it=it,str='train')
        #     self.plot(self.fit(X=self.Xval, Y=self.Yval, mean=mean_val, var=var_val),
        #           it=it,str='val')

    def show_inducing_points_hist(self):

        plt.figure(figsize=(10, 5))
        hist = array(self.inducing_points_hist)

        for m in range(self.model.M):
             plt.plot(self.inducing_points_hist_it,hist[:, m, 0])

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()

        return buf

    def inducing_points(self,it):

        if self.model.D == 1 and hasattr(self.model, 'inducing_points'):
            self.inducing_points_hist.append(self.model.inducing_points)
            self.inducing_points_hist_it.append(it)
            self.plot(
                buf=self.show_inducing_points_hist(),
                it=it,
                str='Inducing points')

    def parameters(self,it):

        with self.writer.as_default():
            tf.summary.scalar("variance", self.model.variance, step=it)
            tf.summary.scalar("gaussian noise", self.model.gaussian_noise, step=it)

            lengthscales = self.model.lengthscales
            inv_lengthscales = self.model.inv_lengthscales
            for d in range(self.model.D):
                tf.summary.scalar("lengthscale {0}".format(d),lengthscales[d], step=it)
                tf.summary.scalar("inv_lengthscale {0}".format(d), inv_lengthscales[d],step=it)

    def _order(self,X,Y,mean,var):
        X = X.numpy()[:, 0]
        Y = Y.numpy()[:, 0]
        mean = mean.numpy()[:, 0]
        var = var.numpy()[:, 0]

        idx = tf.argsort(X)
        return X[idx], Y[idx], mean[idx], var[idx]

    def plot(self,buf,it,str):

        buf.seek(0)

        image = tf.image.decode_png(buf.getvalue(),
                                    channels=4)  # Convert PNG buffer to TF image
        image = tf.expand_dims(image, 0)  # Add the batch dimension

        with self.writer.as_default():
            tf.summary.image(str, image, step=it)

    def fit(self,X,Y,mean,var):

        Y_min = tf.reduce_min(Y).numpy()
        Y_max = tf.reduce_max(Y).numpy()

        X,Y,mean,var = self._order(X,Y,mean,var)

        plt.figure(figsize=(13, 4))
        plt.plot(X,Y, 'r.',markersize=0.2,label='data')
        plt.plot(X, mean, 'b', label='mean')

        ci = 2 * tf.sqrt(var)
        plt.fill_between(X,mean - ci, mean + ci,alpha=0.5, color='b')

        if hasattr(self.model, 'inducing_points'):
            inducing_points = self.model.inducing_points
            for ip in range(self.model.M):
                plt.axvline(x=inducing_points[ip], color='magenta', linestyle='--')

        plt.ylim(Y_min*1.2,Y_max*1.2)
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')

        plt.close()

        return buf

    @tf.function
    def _grad(self,grad_lengthscales,grad_variance,grad_gaussian_noise,grad_inducing_points,it):

        with self.writer.as_default():

            tf.summary.scalar("grad variance",grad_variance, step=it)
            tf.summary.scalar("grad gaussian noise", grad_gaussian_noise,
                              step=it)

            for d in range(self.model.D):
                tf.summary.scalar("grad lengthscale {0}".format(d),grad_lengthscales[d],
                                  step=it)

            for m in range(self.model.M):
                if self.model.D == 1:
                    tf.summary.scalar("grad ip {0}".format(m),
                                      tf.squeeze(grad_inducing_points[m]),
                                      step=it)
                else:
                    tf.summary.scalar("grad norm ip {0}".format(m),
                                      tf.sqrt(tf.reduce_sum(tf.square(grad_inducing_points[0,:]))),
                                      step=it)

    def grad(self,grad,it):

        # based on the order of the variables in self.vars
        grad_lengthscales = grad[0]
        grad_variance = grad[1]
        grad_gaussian_noise = grad[2]
        grad_inducing_points = grad[3]

        self._grad(grad_lengthscales,grad_variance,grad_gaussian_noise,grad_inducing_points,it)


    def write(self,val,str,it):
        with self.writer.as_default():
            tf.summary.scalar(name=str,data=val,step=it)

    def write_if_not_None(self,val,str,it):
        if val is not None:
            self.write(val=val,str=str,it=it)

    def collect(self, it, data_points_used,
                lk=None, ak=None, dk=None, grad=None,
                record=False,
                number_of_batches=None,
                runtime=None):

        if runtime is not None:
            runtime = tf.cast(runtime, dtype=tf.float64)
            self.elapsed_time += runtime

        if record:
            it = tf.cast(it, dtype=tf.int64)
            data_points_used = tf.cast(data_points_used, dtype=tf.int64)

            self.write(val=self.elapsed_time, str="total runtime", it=it)

            self.write_if_not_None(val=lk,str="lowerbound_k (lk)", it=it)
            self.write_if_not_None(val=dk,str="dk", it=it)
            self.write_if_not_None(val=ak,str="ak", it=it)
            self.write_if_not_None(val=number_of_batches,str="number of batches vs it",it=it)

            if it > 1:
                # Skip the first iteration since the graph is created in it
                # Thus, it's usually several orders of magnitude bigger
                # than a normal iteration
                self.write_if_not_None(val=runtime, str="runtime per iteration", it=it)

            # if grad is not None:
            #     self.grad(grad=grad,it=it)

            self.parameters(it)
            # self.inducing_points(it=it)
            self.stats(it=it,data_points_used=data_points_used)
