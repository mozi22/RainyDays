
import tensorflow as tf
import lmbspecialops as sops

def convrelu2(name,inputs, filters, kernel_size, stride, activation=None):

    tmp_y = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=[kernel_size,1],
        strides=[stride,1],
        padding='same',
        name=name+'y',
        activation=activation
    )


    tmp_x = tf.layers.conv2d(
        inputs=tmp_y,
        filters=filters,
        kernel_size=[1,kernel_size],
        strides=[1,stride],
        padding='same',
        activation=activation,
        name=name+'x'
    )

    return tmp_x

def myLeakyRelu(x):
    """Leaky ReLU with leak factor 0.1"""
    # return tf.maximum(0.1*x,x)
    return sops.leaky_relu(x, leak=0.2)


def create_network(input_image,isTrain=True):

    with tf.variable_scope('vae'):

        conv0 = convrelu2(name='conv0', inputs=input_image, filters=16, kernel_size=5, stride=1,activation=myLeakyRelu)
        conv1 = convrelu2(name='conv1', inputs=conv0, filters=16, kernel_size=5, stride=2,activation=myLeakyRelu)

        conv2 = convrelu2(name='conv2', inputs=conv1, filters=32, kernel_size=3, stride=2,activation=myLeakyRelu)

        conv3 = convrelu2(name='conv3', inputs=conv2, filters=64, kernel_size=3, stride=2,activation=myLeakyRelu)
        conv3_1 = convrelu2(name='conv3_1', inputs=conv3, filters=64, kernel_size=3, stride=1,activation=myLeakyRelu)

        conv4 = convrelu2(name='conv4', inputs=conv3_1, filters=128, kernel_size=3, stride=2,activation=myLeakyRelu)
        conv4_1 = convrelu2(name='conv4_1', inputs=conv4, filters=128, kernel_size=3, stride=1,activation=myLeakyRelu)

        dense_slice_shape = conv4_1.get_shape().as_list()

        units = 1
        for i in range(1,len(dense_slice_shape)):
            units *= dense_slice_shape[i]

        dense5 = tf.layers.dense(
                tf.contrib.layers.flatten(tf.slice(conv4_1, [0,0,0,0], dense_slice_shape)),
                units=units,
                activation=myLeakyRelu,
                name='dense5'
        )

        z = 100

        # mean latent vector
        z_mu = tf.layers.dense(dense5,units=z)

        # variance latent vector
        z_sigma = tf.layers.dense(dense5,units=z)

        # normal distribution 
        eps = tf.random_normal(shape=tf.shape(z_sigma),mean=0, stddev=1, dtype=tf.float32)

        # adding up mean, variance with fixed normal distribution
        z = z_mu + tf.sqrt(tf.exp(z_sigma)) * eps

        # reshape [4,100] to [4,1,1,100] to pass it to the conv_transpose
        reshaped_layer = tf.reshape(z,[z.get_shape().as_list()[0],1,1,100])

        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)


        deconv1 = tf.layers.conv2d_transpose(reshaped_layer, 256, [7, 7], strides=(1, 1), padding='valid', kernel_initializer=w_init, bias_initializer=b_init)
        lrelu1 = myLeakyRelu(tf.layers.batch_normalization(deconv1, training=isTrain))

        # 2nd hidden layer
        deconv2 = tf.layers.conv2d_transpose(lrelu1, 256, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        lrelu2 = myLeakyRelu(tf.layers.batch_normalization(deconv2, training=isTrain))

        # 3rd hidden layer
        deconv3 = tf.layers.conv2d_transpose(lrelu2, 128, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        lrelu3 = myLeakyRelu(tf.layers.batch_normalization(deconv3, training=isTrain))

        deconv4 = tf.layers.conv2d_transpose(lrelu3, 128, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        lrelu4 = myLeakyRelu(tf.layers.batch_normalization(deconv4, training=isTrain))

        deconv5 = tf.layers.conv2d_transpose(lrelu4, 64, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        lrelu5 = myLeakyRelu(tf.layers.batch_normalization(deconv5, training=isTrain))

        deconv6 = tf.layers.conv2d_transpose(lrelu5, 3, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)

        o = tf.nn.tanh(deconv6) 

        return o, z_mu, z_sigma