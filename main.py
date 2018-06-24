
import network
import input_pipeline
import tensorflow as tf
import losses_helper

dataset = input_pipeline.parse()
iterator = dataset.make_one_shot_iterator()

# get input
input_image, resulting_image = iterator.get_next()



# make prediction
prediction, z_mu, z_sigma = network.create_network(input_image)

####### define losses

# resize input_image for calculating reconstruction loss on a slightly smaller image.
input_image_resized = tf.image.resize_images(input_image,[224,224],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

loss_recon = losses_helper.reconstruction_loss(prediction,input_image_resized)
loss_kl = losses_helper.KL_divergence_loss(z_mu,z_sigma)


