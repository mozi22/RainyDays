
import tensorflow as tf
import lmbspecialops as sops

def reconstruction_loss(prediction,gt):

	with tf.variable_scope('reconstruction_loss'):
		rec_loss = tf.reduce_mean(tf.reduce_sum((gt - prediction)**2, axis=[1, 2, 3]))
		# rec_loss = -tf.reduce_sum(gt * tf.log(1e-8 + prediction) + (1-gt) * tf.log(1e-8 + 1 - prediction), axis=[1, 2, 3])
		recon_loss = sops.replace_nonfinite(rec_loss)

		# recon_loss = tf.reduce_mean(recon_loss)

	return recon_loss

def KL_divergence_loss(z_mu,z_log_sigma_sq):

	with tf.variable_scope('kl_loss'):

		latent_loss = -tf.reduce_mean(0.5 * tf.reduce_sum(1 + z_log_sigma_sq - z_mu**2 - tf.exp(z_log_sigma_sq), axis=1))
		latent_loss = sops.replace_nonfinite(latent_loss * 3000)

	return latent_loss