
import tensorflow as tf
import lmbspecialops as sops
import numpy as np

def conv(name,inputs, filters, kernel_size, stride, activation=myLeakyRelu):

    layer = tf.layers.conv2d(inputs=inputs, 
                            filters=filters, 
                            kernel_size=kernel_size, 
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            strides=stride, 
                            kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001),
                            activation=myLeakyRelu,
                            name=name)

    return layer

def conv_transpose(name,inputs, filters, kernel_size, stride):

    layer = tf.layers.conv2d_transpose(inputs=inputs, 
                                       filters=filters, 
                                       kernel_size=kernel_size, 
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                       strides=stride, 
                                       kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001),
                                       activation=myLeakyRelu,
                                       name=name)

    return layer


def myLeakyRelu(x):
    """Leaky ReLU with leak factor 0.1"""
    # return tf.maximum(0.1*x,x)
    return sops.leaky_relu(x, leak=0.2)

def tanh(x) :
    return tf.tanh(x)


def encoder_decoder(input_image,gan_enabled=False):

    with tf.variable_scope('vae'):

        conv1 = conv(name='conv1', inputs=input_image, filters=64, kernel_size=7, stride=1)
        conv2 = conv(name='conv2', inputs=conv1, filters=128, kernel_size=3, stride=2)
        conv3 = conv(name='conv3', inputs=conv2, filters=256, kernel_size=3, stride=2)

        sigma = 1.0
        z_random = tf.random_normal(shape=tf.shape(conv3), mean=0.0, stddev=1.0, dtype=tf.float32)
        
        latent_space = mu + sigma * z_random

        conv_tran3 = conv_transpose(name='conv3_transpose', inputs=latent_space, filters=128, kernel_size=3, stride=2)
        conv_tran2 = conv_transpose(name='conv2_transpose', inputs=conv_tran3, filters=64, kernel_size=3, stride=2)
        conv_tran1 = conv_transpose(name='conv1_transpose', inputs=conv_tran2, filters=3, kernel_size=1, stride=1, activation=tanh)


        return latent_space, conv_tran1

