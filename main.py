
from tensorflow.contrib.tensorboard.plugins import projector
import network
import input_pipeline
import tensorflow as tf
import numpy as np
import time
import losses_helper
import os
from datetime import datetime


# ################## embedding ##################
# def generate_embedding(embedding_var):

# 	# Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
# 	config = projector.ProjectorConfig()

# 	# You can add multiple embeddings. Here we add only one.
# 	embedding = config.embeddings.add()
# 	embedding.tensor_name = embedding_var.name
# 	# Link this tensor to its metadata file (e.g. labels).
# 	embedding.metadata_path = os.path.join(ckpt_folder, 'metadata.tsv')

# 	# Use the same LOG_DIR where you stored your checkpoint.

# 	# The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
# 	# read this file during startup.
# 	projector.visualize_embeddings(summary_writer, config)
# ################## embedding ##################


recon_loss_weight = 100
kl_loss_weight = 0.1




dataset = input_pipeline.parse()
iterator = dataset.make_initializable_iterator()

discriminator_on = False
ckpt_folder = './ckpt/working_copy'

####### get input #######
input_image, resulting_image = iterator.get_next()

####### make prediction #######
latent_space, prediction = network.encoder_decoder(input_image,discriminator_on)
####### define losses #######
# resize input_image for calculating reconstruction loss on a slightly smaller image.
# resulting_image_resized = tf.image.resize_images(input_image,[25,64],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

loss_recon = losses_helper.reconstruction_loss_l1(prediction,input_image)
loss_kl = losses_helper.KL_divergence_loss(latent_space)

total_vae_loss = loss_recon * recon_loss_weight + loss_kl * kl_loss_weight




MAX_ITERATIONS = 20000
alternate_global_step = tf.placeholder(tf.int32)

global_step = tf.get_variable(
    'global_step', [],
    initializer=tf.constant_initializer(0), trainable=False)

learning_rate = tf.train.polynomial_decay(0.0001, alternate_global_step,
                                          MAX_ITERATIONS, 0.000001,
                                          power=3)



t_vars = tf.trainable_variables()


g_vars = tf.trainable_variables()
opt = tf.train.AdamOptimizer(learning_rate,beta1=0.5, beta2=0.999)
grads = opt.compute_gradients(total_vae_loss,var_list=g_vars)
apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
 

variable_averages_g = tf.train.ExponentialMovingAverage(
    0.9999, global_step)
variables_averages_op_g = variable_averages_g.apply(g_vars)
train_op = tf.group(apply_gradient_op, variables_averages_op_g)




####### write summaries #######

train_summaries = []

train_summaries.append(tf.summary.scalar('recon_loss',loss_recon))


train_summaries.append(tf.summary.histogram('prediction',prediction))
train_summaries.append(tf.summary.histogram('gt',input_image))
train_summaries.append(tf.summary.histogram('z_latent',latent_space))
train_summaries.append(tf.summary.scalar('kl_loss',loss_kl))
train_summaries.append(tf.summary.scalar('learning_rate',learning_rate))
train_summaries.append(tf.summary.scalar('total_vae_loss',total_vae_loss))
train_summaries.append(tf.summary.image('input_image',input_image))

# train_summaries.append(tf.summary.image('resulting_image',resulting_image_resized))
train_summaries.append(tf.summary.image('predicted_image',prediction))

for grad, var in grads:
    if grad is not None:
        train_summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

for var in tf.trainable_variables():
    train_summaries.append(tf.summary.histogram(var.op.name, var))


summary_op = tf.summary.merge(train_summaries)



####### create savers and tensorboard #######
saver = tf.train.Saver(tf.global_variables())

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
sess.run(iterator.initializer)
# saver.restore(sess,tf.train.latest_checkpoint(ckpt_folder+'/'))

loop_start = tf.train.global_step(sess, global_step)
loop_stop = loop_start + MAX_ITERATIONS

summary_writer = tf.summary.FileWriter(ckpt_folder, sess.graph)
# generate_embedding(z_mu)
# generate_embedding(z_sigma)
# generate_embedding(z_latent)

first_iteration = True
iteration = 0

folder_name = ckpt_folder.split('/')[-1]
for step in range(loop_start,loop_stop+1):


	# generator_iterations = 1

	# for i in range(generator_iterations):
	_ , loss = sess.run([train_op,total_vae_loss],feed_dict={
			alternate_global_step: iteration
	})




	assert not np.isnan(loss), 'Model diverged with loss = NaN'

	format_str = ('%s: step %d, loss = %.15f, folder = %s')
	print((format_str % (datetime.now(),step, loss, folder_name)))


	if step % 500 == 0:
		summmary = sess.run(summary_op, feed_dict={
			alternate_global_step: iteration
		})
		summary_writer.add_summary(summmary,step)

	# Save the model checkpoint periodically.
	if step % 1000 == 0:
		checkpoint_path = os.path.join(ckpt_folder, 'model.ckpt')
		saver.save(sess, checkpoint_path, global_step=step)

	iteration += 1

summary_writer.close()






