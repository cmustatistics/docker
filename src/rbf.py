#########################################
# Disclaimer: The functions in this file are based on the
# RBF implementation provided by GPflow 2.0
#########################################

import tensorflow as tf


@tf.function
def K(inv_lengthscales, variance, X, X2):
    X = X * inv_lengthscales
    X2 = X2 * inv_lengthscales

    Xs = tf.reduce_sum(tf.square(X), axis=-1)
    X2s = tf.reduce_sum(tf.square(X2), axis=-1)
    dist = -2 * tf.tensordot(X, X2, [[-1], [-1]])

    flatres = tf.add(tf.reshape(Xs, [-1, 1]), tf.reshape(X2s, [1, -1]))
    dist += tf.reshape(flatres, tf.concat([tf.shape(Xs), tf.shape(X2s)], 0))

    return K_r2(dist, variance)


@tf.function
def dist(X):
    Xs = tf.reduce_sum(tf.square(X), axis=-1, keepdims=True)
    dist = -2 * tf.matmul(X, X, transpose_b=True)
    dist += Xs + tf.linalg.adjoint(Xs)
    return dist


@tf.function
def Ksym(inv_lengthscales, variance, X):
    X = X * inv_lengthscales
    return K_r2(dist(X), variance)


@tf.function
def K_r2(r2, variance):
    r2 = tf.maximum(r2, 1e-36)
    return variance * tf.exp(-0.5 * r2)


@tf.function
def K_diag(variance, X):
    return tf.fill(tf.shape(X)[:-1], tf.squeeze(variance))
