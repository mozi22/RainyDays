
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


def gan_loss(fake_flow_d,real_flow_d,conv_real,conv_fake,weight=1):

  EPS = 1e-12

  with tf.variable_scope('generator_loss'):
    g_total_loss = sops.replace_nonfinite(tf.reduce_mean(-tf.log(fake_flow_d + EPS)))
    # g_total_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=conv4_fake,labels=tf.ones_like(conv4_real)))

  with tf.variable_scope('discriminator_loss'):
    d_total_loss = sops.replace_nonfinite(tf.reduce_mean(-(tf.log(real_flow_d + EPS) + tf.log(1 - fake_flow_d + EPS))))

    # d_total_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=conv4_real,labels=tf.ones_like(conv4_real)))
    # d_total_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=conv4_fake,labels=tf.zeros_like(conv4_real)))
    # d_total_loss = d_total_loss_fake + d_total_loss_real
    # d_total_loss = sops.replace_nonfinite(d_total_loss)
    # feature_matching_loss = endpoint_loss(conv_real,conv_fake,weight=weight + 10,scope='feature_matching_loss')

    # tf.add_to_collection('disc_loss',feature_matching_loss)
    tf.add_to_collection('disc_loss',d_total_loss)

    # tf.summary.scalar('disc_loss'+summary_type,d_total_loss)
    # tf.summary.scalar('feature_matching_loss',feature_matching_loss)

  return g_total_loss, d_total_loss