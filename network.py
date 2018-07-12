
import tensorflow as tf
import lmbspecialops as sops
import numpy as np

def myLeakyRelu(x):
    """Leaky ReLU with leak factor 0.1"""
    # return tf.maximum(0.1*x,x)
    return sops.leaky_relu(x, leak=0.2)


def conv(name,inputs, filters, kernel_size, stride, activation=myLeakyRelu, pad=0):

    inputs = tf.pad(inputs, [[0,0], [pad, pad], [pad, pad], [0,0]])

    layer = tf.layers.conv2d(inputs=inputs, 
                            filters=filters, 
                            kernel_size=kernel_size, 
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            strides=stride, 
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
                            activation=myLeakyRelu,
                            name=name)

    return layer

def conv_transpose(name,inputs, filters, kernel_size, stride, activation=myLeakyRelu):

    layer = tf.layers.conv2d_transpose(inputs=inputs, 
                                       filters=filters, 
                                       kernel_size=kernel_size,
                                       padding='SAME',
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                       strides=stride, 
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
                                       activation=myLeakyRelu,
                                       name=name)

    return layer



def resblock(name,inputs, filters, kernel_size=3, stride=1, pad=1) :

    with tf.variable_scope(name) :
        with tf.variable_scope('res1') :

            x = tf.pad(inputs, [[0, 0], [pad, pad], [pad, pad], [0, 0]])

            x = tf.layers.conv2d(inputs=x, filters=filters, kernel_size=kernel_size,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 strides=stride, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001))

            x = instance_norm(x, 'res1_instance')
            x = myLeakyRelu(x)

        with tf.variable_scope('res2') :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])

            x = tf.layers.conv2d(inputs=x, filters=filters, kernel_size=kernel_size,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 strides=stride, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001))

            x = instance_norm(x, 'res2_instance')

        return x + inputs


def encoder_decoder(input_image):

    with tf.variable_scope('vae'):

        conv1 = conv(name='conv1', inputs=input_image, filters=64, kernel_size=7, stride=1, pad=3)
        conv2 = conv(name='conv2', inputs=conv1, filters=128, kernel_size=3, stride=2, pad=1)
        conv3 = conv(name='conv3', inputs=conv2, filters=256, kernel_size=3, stride=2, pad=1)

        res_b1 = resblock(name='resb_1', inputs=conv3, filters=256, kernel_size=3, stride=1, pad=1)
        res_b2 = resblock(name='resb_2', inputs=res_b1, filters=256, kernel_size=3, stride=1, pad=1)
        res_b3 = resblock(name='resb_3', inputs=res_b2, filters=256, kernel_size=3, stride=1, pad=1)

        ######################## creating latent space ###############################

        sigma = 1.0
        z_random = tf.random_normal(shape=tf.shape(res_b3), mean=0.0, stddev=1.0, dtype=tf.float32)
        
        latent_space = res_b3 + sigma * z_random

        ######################## creating latent space ###############################


        de_res_b1 = resblock(name='de_resb_1', inputs=latent_space, filters=256, kernel_size=3, stride=1, pad=1)
        de_res_b2 = resblock(name='de_resb_2', inputs=de_res_b1, filters=256, kernel_size=3, stride=1, pad=1)
        de_res_b3 = resblock(name='de_resb_3', inputs=de_res_b2, filters=256, kernel_size=3, stride=1, pad=1)

        conv_tran3 = conv_transpose(name='conv3_transpose', inputs=de_res_b3, filters=128, kernel_size=3, stride=2)
        conv_tran2 = conv_transpose(name='conv2_transpose', inputs=conv_tran3, filters=64, kernel_size=3, stride=2)
        conv_tran1 = conv_transpose(name='conv1_transpose', inputs=conv_tran2, filters=3, kernel_size=1, stride=1, activation=tf.nn.sigmoid)


        return latent_space, conv_tran1


def discriminator(x,reuse=False,scope="discriminator"):
  with tf.variable_scope(scope, reuse=reuse):
    conv1 = conv(name='disc_conv1',inputs=x,filters=64,kernel_size=3,stride=2,pad=1)
    conv2 = conv(name='disc_conv2',inputs=conv1,filters=128,kernel_size=3,stride=2,pad=1)
    conv3 = conv(name='disc_conv3',inputs=conv2,filters=256,kernel_size=3,stride=2,pad=1)
    conv4 = conv(name='disc_conv4',inputs=conv3,filters=512,kernel_size=3,stride=2,pad=1)
    conv5 = conv(name='disc_conv5',inputs=conv4,filters=1024,kernel_size=3,stride=2,pad=1)
    conv6 = conv(name='disc_conv6',inputs=conv5,filters=2048,kernel_size=3,stride=2,pad=1)
    conv7 = conv(name='disc_conv7',inputs=conv6,filters=1,kernel_size=1,stride=1,pad=0,activation=None)

    return conv7

def instance_norm(x, scope='instance') :
    return tf.contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)