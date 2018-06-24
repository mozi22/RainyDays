
def reconstruction_loss(prediction,gt):

    with tf.variable_scope('reconstruction_loss'):
	    epsilon = 1e-10
	    recon_loss = -tf.reduce_sum(
	        gt * tf.log(epsilon+prediction) + (1-gt) * tf.log(epsilon+1-prediction), 
	        axis=1
	    )

	    recon_loss = tf.reduce_mean(recon_loss)

	return recon_loss

def KL_divergence_loss(z_mu,z_log_sigma_sq):

    with tf.variable_scope('kl_loss'):

	    latent_loss = -0.5 * tf.reduce_sum(
	        1 + z_log_sigma_sq - tf.square(z_mu) - tf.exp(z_log_sigma_sq), axis=1)
	    latent_loss = tf.reduce_mean(latent_loss)

	    total_loss = tf.reduce_mean(recon_loss + latent_loss)

    return total_loss