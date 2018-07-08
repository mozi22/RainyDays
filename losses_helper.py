
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
