
import tensorflow as tf
import lmbspecialops as sops
import numpy as np
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

def predict_final_image(inp):
    """Generates a tensor for optical flow prediction
    
    inp: Tensor

    predict_confidence: bool
        If True the output tensor has 4 channels instead of 2.
        The last two channels are the x and y flow confidence.
    """

    

    tmp = tf.layers.conv2d(
        inputs=inp,
        filters=24,
        kernel_size=3,
        strides=1,
        padding='same',
        name='conv1_pred',
        activation=myLeakyRelu
    )

    output = tf.layers.conv2d(
        inputs=tmp,
        filters=3,
        kernel_size=3,
        strides=1,
        padding='same',
        name='conv2_pred',
        activation=None
    )

    
    return output
def _refine(inp, num_outputs, upsampled_prediction=None, features_direct=None,name=None):
    """ Generates the concatenation of 
         - the previous features used to compute the flow/depth
         - the upsampled previous flow/depth
         - the direct features that already have the correct resolution

    inp: Tensor
        The features that have been used before to compute flow/depth

    num_outputs: int 
        number of outputs for the upconvolution of 'features'

    upsampled_prediction: Tensor
        The upsampled flow/depth prediction

    features_direct: Tensor
        The direct features which already have the spatial output resolution
    """
    upsampled_features = tf.layers.conv2d_transpose(
        inputs=inp,
        filters=num_outputs,
        kernel_size=4,
        strides=2,
        padding='same',
        activation=myLeakyRelu,
        name="upconv"
    )




    inputs = [upsampled_features, features_direct, upsampled_prediction]
    concat_inputs = [ x for x in inputs if not x is None ]

    return tf.concat(concat_inputs, axis=3)

def myLeakyRelu(x):
    """Leaky ReLU with leak factor 0.1"""
    # return tf.maximum(0.1*x,x)
    return sops.leaky_relu(x, leak=0.2)


def create_network(input_image,gan_enabled=False):

    with tf.variable_scope('vae'):

        conv1 = convrelu2(name='conv1', inputs=input_image, filters=64, kernel_size=5, stride=2,activation=myLeakyRelu)

        if gan_enabled == True:
            conv1 = tf.layers.dropout(conv1)

        conv2 = convrelu2(name='conv2', inputs=conv1, filters=128, kernel_size=3, stride=2,activation=myLeakyRelu)

        if gan_enabled == True:
            conv2 = tf.layers.dropout(conv2)

        conv3 = convrelu2(name='conv3', inputs=conv2, filters=256, kernel_size=3, stride=2,activation=myLeakyRelu)

        if gan_enabled == True:
            conv3 = tf.layers.dropout(conv3)

        # conv4 = convrelu2(name='conv4', inputs=conv3, filters=128, kernel_size=2, stride=2,activation=myLeakyRelu)

        # if gan_enabled == True:
        #     conv4 = tf.layers.dropout(conv4)

        # conv5 = convrelu2(name='conv5', inputs=conv4, filters=256, kernel_size=2, stride=2,activation=myLeakyRelu)

        # if gan_enabled == True:
        #     conv5 = tf.layers.dropout(conv5)

        # conv6 = convrelu2(name='conv6', inputs=conv5, filters=256, kernel_size=2, stride=2,activation=myLeakyRelu)

        # if gan_enabled == True:
        #     conv6 = tf.layers.dropout(conv6)

        # print(conv6)

        dense_slice_shape = conv3.get_shape().as_list()

        # dense_slice_shape[-1] = 96

        # units = 1
        # for i in range(1,len(dense_slice_shape)):
        #     units *= dense_slice_shape[i]

        # dense5 = tf.layers.dense(
        #         tf.contrib.layers.flatten(tf.slice(conv3, [0,0,0,0], dense_slice_shape)),
        #         units=units,
        #         activation=None,
        #         name='dense5'
        # )
        print(conv3)
        dense5 = tf.contrib.layers.flatten(conv3)
        print(dense5)
        z = 2048


        # mean latent vector
        z_mu = tf.layers.dense(dense5,units=z)

        # variance latent vector
        z_sigma = tf.layers.dense(dense5,units=z)

        # normal distribution 
        eps = tf.random_normal(shape=tf.shape(z_sigma),mean=0, stddev=1, dtype=tf.float32)

        # adding up mean, variance with fixed normal distribution
        z_latent = z_mu + (z_sigma * eps)

        full_units_layer = tf.contrib.layers.fully_connected(z_latent,dense_slice_shape[1]*dense_slice_shape[2]*dense_slice_shape[3])

        # reshape [4,100] to [4,1,1,100] to pass it to the conv_transpose
        reshaped_layer = tf.reshape(full_units_layer,[full_units_layer.get_shape().as_list()[0],8,8,256])

        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(0.0)


        # 1rd hidden layer
        deconv1 = tf.layers.conv2d_transpose(reshaped_layer, 256, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        # lrelu1 = myLeakyRelu(tf.layers.batch_normalization(deconv1, training=True))
        if gan_enabled == True:
            deconv1 = tf.layers.dropout(deconv1)

        lrelu1 = myLeakyRelu(deconv1)

        # 2nd hidden layer
        deconv2 = tf.layers.conv2d_transpose(lrelu1, 128, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        # lrelu2 = myLeakyRelu(tf.layers.batch_normalization(deconv2, training=True))
        if gan_enabled == True:
            deconv2 = tf.layers.dropout(deconv2)
    
        lrelu2 = myLeakyRelu(deconv2)

        # 3rd hidden layer
        deconv3 = tf.layers.conv2d_transpose(lrelu2, 32, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        # lrelu3 = myLeakyRelu(tf.layers.batch_normalization(deconv3, training=True))
        if gan_enabled == True:
            deconv3 = tf.layers.dropout(deconv3)
    
        lrelu3 = myLeakyRelu(deconv3)

        # # 4rd hidden layer
        # deconv4 = tf.layers.conv2d_transpose(lrelu3, 128, [3, 3], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        # # lrelu4 = myLeakyRelu(tf.layers.batch_normalization(deconv4, training=True))
        # if gan_enabled == True:
        #     deconv4 = tf.layers.dropout(deconv4)
    
        # lrelu4 = myLeakyRelu(deconv4)

        # # 5rd hidden layer
        # deconv5 = tf.layers.conv2d_transpose(lrelu4, 128, [3, 3], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        # # lrelu5 = myLeakyRelu(tf.layers.batch_normalization(deconv5, training=True))
        # if gan_enabled == True:
        #     deconv5 = tf.layers.dropout(deconv5)
    
        # lrelu5 = myLeakyRelu(deconv5)

        # deconv6 = tf.layers.conv2d_transpose(lrelu5, 64, [3, 3], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        # # lrelu6 = myLeakyRelu(tf.layers.batch_normalization(deconv6, training=True))
        # if gan_enabled == True:
        #     deconv6 = tf.layers.dropout(deconv6)
    
        # lrelu6 = myLeakyRelu(deconv6)
        # lrelu6 = tf.nn.sigmoid(deconv6)

        prediction = predict_final_image(lrelu3)
        print(prediction)

        return prediction, z_mu, z_sigma, z_latent



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