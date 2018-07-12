
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
                            activation=activation,
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
                                       activation=activation,
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



''' 
    Step 1: passes both the domain images individually to encoder.
    Step 2: concats [1,H,W,C] images from both domains into [2,H,W,C]
    Step 3: pass them through a conv layer to learn shared weights on that layer
    Step 4: pass the result through the latent space to learned a shared space for both domains
    Step 5: pass through a deconv layer for sharing weights among both domains 
    Step 6: pass through the generator. Note we still have [2,H,W,C]. We want to pass them through
            generator together because, we want to learn to generate both kinds of imgs from the 
            shared latent space. And also the reconstructed images (the input images).
    Step 7: Split out the results now. We'll get ab,aa,ba,bb results. Considering the domains are called A & B.
'''
def perform_image_translation(domainA,domainB):

    print('translation')
    print(domainA)
    print(domainB)
    enc_domain_A = encoder(domainA,'encoderA')  # Step 1
    enc_domain_B = encoder(domainB,'encoderB')  # Step 1
    # encoding 
    encoded_imgs = tf.concat([enc_domain_A,enc_domain_B],axis=0)  # Step 2
    print(encoded_imgs)
    encoded_imgs = shared_res_layer_encoder(encoded_imgs)   # Step 3
    print(encoded_imgs)


    # latent space
    enc_shared_latent_space = shared_latent_space(encoded_imgs) # Step 4
    print(enc_shared_latent_space)

    decoded_imgs = shared_res_layer_decoder(enc_shared_latent_space) # Step 5
    print(decoded_imgs)

    # decoding
    decoded_imgsA = decoder(decoded_imgs,'decoderA') # Step 6
    decoded_imgsB = decoder(decoded_imgs,'decoderB') # Step 6

    print(decoded_imgsA)
    print(decoded_imgsB)

    # Ab means A to B and same for others.
    input_Aa, input_Ab = tf.split(decoded_imgsA, 2, axis=0) # Step 7
    input_Ba, input_Bb = tf.split(decoded_imgsB, 2, axis=0) # Step 7

    print(input_Aa)
    print(input_Ab)
    print(input_Ba)
    print(input_Bb)

    return input_Aa, input_Ab, input_Ba, input_Bb, enc_shared_latent_space



def generate_atob(input):
  enc = encoder(input,'encoderA',reuse=True)
  enc_sh = shared_res_layer_encoder(enc,reuse=True)
  latent_space = shared_latent_space(enc_sh,reuse=True)
  dec_sh = shared_res_layer_decoder(latent_space,reuse=True)
  dec = decoder(dec_sh,'decoderB',reuse=True)
  return dec, latent_space

def generate_btoa(input):
  enc = encoder(input,'encoderB',reuse=True)
  enc_sh = shared_res_layer_encoder(enc,reuse=True)
  latent_space = shared_latent_space(enc_sh,reuse=True)
  dec_sh = shared_res_layer_decoder(latent_space,reuse=True)
  dec = decoder(dec_sh,'decoderA',reuse=True)
  return dec, latent_space



def encoder(input_image,scope='encoder',reuse=False):

    with tf.variable_scope(scope,reuse=reuse):

        conv1 = conv(name='conv1', inputs=input_image, filters=64, kernel_size=7, stride=1, pad=3)
        conv2 = conv(name='conv2', inputs=conv1, filters=128, kernel_size=3, stride=2, pad=1)
        conv3 = conv(name='conv3', inputs=conv2, filters=256, kernel_size=3, stride=2, pad=1)

        res_b1 = resblock(name='resb_1', inputs=conv3, filters=256, kernel_size=3, stride=1, pad=1)
        res_b2 = resblock(name='resb_2', inputs=res_b1, filters=256, kernel_size=3, stride=1, pad=1)
        res_b3 = resblock(name='resb_3', inputs=res_b2, filters=256, kernel_size=3, stride=1, pad=1)

        return res_b3


def shared_res_layer_encoder(input,scope='shared_encoder',reuse=False):

    with tf.variable_scope(scope,reuse=reuse):
        shared_resb_encoder = resblock(name='shared_resb_encoder', inputs=input, filters=256, kernel_size=3, stride=1, pad=1)
    return shared_resb_encoder

def shared_res_layer_decoder(input,scope='shared_decoder',reuse=False):

    with tf.variable_scope(scope,reuse=reuse):
        shared_resb_decoder = resblock(name='shared_resb_decoder', inputs=input, filters=256, kernel_size=3, stride=1, pad=1)
    return shared_resb_decoder


def shared_latent_space(input,scope='shared_latent_space',reuse=False):

    with tf.variable_scope(scope,reuse=reuse):
        sigma = 1.0
        z_random = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=1.0, dtype=tf.float32)
        
        latent_space = input + sigma * z_random

        return latent_space

def decoder(input,scope='decoder',reuse=False):

    with tf.variable_scope(scope,reuse=reuse):
        de_res_b1 = resblock(name='de_resb_1', inputs=input, filters=256, kernel_size=3, stride=1, pad=1)
        de_res_b2 = resblock(name='de_resb_2', inputs=de_res_b1, filters=256, kernel_size=3, stride=1, pad=1)
        de_res_b3 = resblock(name='de_resb_3', inputs=de_res_b2, filters=256, kernel_size=3, stride=1, pad=1)

        conv_tran3 = conv_transpose(name='conv3_transpose', inputs=de_res_b3, filters=128, kernel_size=3, stride=2)
        conv_tran2 = conv_transpose(name='conv2_transpose', inputs=conv_tran3, filters=64, kernel_size=3, stride=2)
        conv_tran1 = conv_transpose(name='conv1_transpose', inputs=conv_tran2, filters=3, kernel_size=1, stride=1, activation=tf.nn.tanh)

        return conv_tran1


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



def discriminate_real(x_A, x_B):
    real_A_logit = discriminator(x_A, scope="discriminator_A")
    real_B_logit = discriminator(x_B, scope="discriminator_B")

    return real_A_logit, real_B_logit

def discriminate_fake(x_ba, x_ab):
    fake_A_logit = discriminator(x_ba, reuse=True, scope="discriminator_A")
    fake_B_logit = discriminator(x_ab, reuse=True, scope="discriminator_B")

    return fake_A_logit, fake_B_logit


def instance_norm(x, scope='instance') :
    return tf.contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)