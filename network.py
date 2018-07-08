
import tensorflow as tf
import lmbspecialops as sops
import numpy as np

def convrelu2(name,inputs, filters, kernel_size, stride):

    layer = tf.layers.conv2d(inputs=inputs, 
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


def create_network(input_image,gan_enabled=False):

    with tf.variable_scope('vae'):

        conv1 = convrelu2(name='conv1', inputs=input_image, filters=64, kernel_size=7, stride=1)

        if gan_enabled == True:
            conv1 = tf.layers.dropout(conv1)

        conv2 = convrelu2(name='conv2', inputs=conv1, filters=128, kernel_size=3, stride=2)

        if gan_enabled == True:
            conv2 = tf.layers.dropout(conv2)

        conv3 = convrelu2(name='conv3', inputs=conv2, filters=256, kernel_size=3, stride=2)

        if gan_enabled == True:
            conv3 = tf.layers.dropout(conv3)






def discriminator(input, is_train=True, reuse=False):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        conv0 = convrelu2(name='conv0', inputs=input, filters=32, kernel_size=5, stride=2,activation=None)
        # conv0 = tf.layers.batch_normalization(conv0,training=is_train)
        conv0 =myLeakyRelu(conv0)

        conv1 = convrelu2(name='conv1', inputs=conv0, filters=64, kernel_size=3, stride=2,activation=None)
        # conv1 = tf.layers.batch_normalization(conv1,training=is_train)
        conv1 =myLeakyRelu(conv1)

        conv2 = convrelu2(name='conv2', inputs=conv1, filters=128, kernel_size=3, stride=2,activation=None)
        # conv2 = tf.layers.batch_normalization(conv2,training=is_train)
        conv2 =myLeakyRelu(conv2)

        conv3 = convrelu2(name='conv3', inputs=conv2, filters=256, kernel_size=3, stride=2,activation=None)
        # conv3 = tf.layers.batch_normalization(conv3,training=is_train)
        conv3 =myLeakyRelu(conv3)

        conv4 = convrelu2(name='conv4', inputs=conv3, filters=512, kernel_size=3, stride=2,activation=None)
        # conv4 = tf.layers.batch_normalization(conv4,training=is_train)
        conv4 =myLeakyRelu(conv4)

        dim = int(np.prod(conv4.get_shape()[1:]))
        fc1 = tf.reshape(conv4, shape=[-1, dim], name='fc1')
      
        
        w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

        # wgan just get rid of the sigmoid
        logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')
        # dcgan
        acted_out = tf.nn.sigmoid(logits)

        # dcgan
        return acted_out, conv2 #, acted_out