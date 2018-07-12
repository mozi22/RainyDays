
import tensorflow as tf
import lmbspecialops as sops

def reconstruction_loss_l1(prediction,gt):

	with tf.variable_scope('reconstruction_loss_l1'):
		loss = tf.reduce_mean(tf.abs(prediction - gt))

	return loss

def KL_divergence_loss(z_mu):

	with tf.variable_scope('kl_loss'):
		# KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, axis = -1)
		# loss = tf.reduce_mean(KL_divergence)
		mu_2 = tf.square(z_mu)
		loss = tf.reduce_mean(mu_2)

	return loss

def generator_loss(fake) :

	with tf.variable_scope('generator_loss'):
		fake_labels = tf.ones_like(fake)
		loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_labels, logits=fake))

	return loss

def discriminator_loss(real, fake) :

	with tf.variable_scope('discriminator_loss'):
		real_labels = tf.ones_like(real)
		fake_labels = tf.zeros_like(fake)

		real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=real_labels, logits=real))
		fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_labels, logits=fake))

		loss = real_loss + fake_loss

	return loss