
import network
import input_pipeline
import tensorflow as tf
import numpy as np
import time
import losses_helper
import os
from datetime import datetime

dataset = input_pipeline.parse()
iterator = dataset.make_initializable_iterator()

discriminator_on = False
ckpt_folder = './ckpt/bigger_latent_space'

####### get input #######
input_image, resulting_image = iterator.get_next()



####### make prediction #######
prediction, z_mu, z_sigma = network.create_network(input_image)

####### define losses #######

# resize input_image for calculating reconstruction loss on a slightly smaller image.
input_image_resized = tf.image.resize_images(input_image,[224,224],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

loss_recon = losses_helper.reconstruction_loss(prediction,input_image_resized)
loss_kl = losses_helper.KL_divergence_loss(z_mu,z_sigma)

loss_recon = tf.reduce_mean(loss_recon)
loss_kl = tf.reduce_mean(loss_kl)

total_loss = tf.reduce_mean(loss_recon + loss_kl)

####### gan loss #######

real_d, conv_real  = network.discriminator(input_image_resized,True)
fake_d, conv_fake = network.discriminator(prediction,True,True)

g_total_loss, d_total_loss = losses_helper.gan_loss(fake_d,real_d,conv_real,conv_fake)

total_loss = total_loss * 100 + g_total_loss

####### initialize optimizer #######

MAX_ITERATIONS = 50000
alternate_global_step = tf.placeholder(tf.int32)

global_step = tf.get_variable(
    'global_step', [],
    initializer=tf.constant_initializer(0), trainable=False)

learning_rate = tf.train.polynomial_decay(0.0001, alternate_global_step,
                                          MAX_ITERATIONS, 0.000001,
                                          power=3)

if discriminator_on == True:
	learning_rate_d = tf.train.polynomial_decay(0.0001, alternate_global_step,
	                                          MAX_ITERATIONS, 0.000001,
	                                          power=3)


t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'dis' in var.name]
g_vars = [var for var in t_vars if 'vae' in var.name]


opt = tf.train.AdamOptimizer(learning_rate)
grads = opt.compute_gradients(total_loss,var_list=g_vars)
apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

if discriminator_on == True:
	opt_d = tf.train.AdamOptimizer(learning_rate_d)
	d_grads = opt_d.compute_gradients(d_total_loss,var_list=d_vars)
	apply_gradient_op_d = opt.apply_gradients(d_grads, global_step=global_step)

# Track the moving averages of all trainable variables.
variable_averages = tf.train.ExponentialMovingAverage(
    0.9999, global_step)
variables_averages_op = variable_averages.apply(tf.trainable_variables())

# Group all updates to into a single train op.
train_op = tf.group(apply_gradient_op, variables_averages_op)

if discriminator_on == True:
	train_op_d = tf.group(apply_gradient_op_d, variables_averages_op)



####### write summaries #######

train_summaries = []

train_summaries.append(tf.summary.scalar('recon_loss',loss_recon))
train_summaries.append(tf.summary.scalar('g_loss',g_total_loss))
train_summaries.append(tf.summary.scalar('d_loss',d_total_loss))
train_summaries.append(tf.summary.histogram('prediction',prediction))
train_summaries.append(tf.summary.histogram('gt',input_image_resized))
train_summaries.append(tf.summary.scalar('kl_loss',loss_kl))
train_summaries.append(tf.summary.scalar('total_loss',total_loss))
train_summaries.append(tf.summary.image('input_image',input_image_resized))
train_summaries.append(tf.summary.image('resulting_image',resulting_image))
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


first_iteration = True
iteration = 0
for step in range(loop_start,loop_stop):


	_ , loss = sess.run([train_op,total_loss],feed_dict={
			alternate_global_step: iteration
	})

	if discriminator_on == True:
		_ , loss_d = sess.run([train_op_d,d_total_loss],feed_dict={
				alternate_global_step: iteration
		})



	assert not np.isnan(loss), 'Model diverged with loss = NaN'

	format_str = ('%s: step %d, loss = %.15f')
	print((format_str % (datetime.now(),step, loss)))

	if discriminator_on == True:
		format_str = ('%s: step %d, loss_d = %.15f')
		print((format_str % (datetime.now(),step, loss_d)))
		print('')

	if step % 100 == 0:
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