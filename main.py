
import network
import input_pipeline
import tensorflow as tf
import numpy as np
import time
import losses_helper
import os
from datetime import datetime


recon_loss_weight = 100
kl_loss_weight = 0.1
GAN_weight = 10


disc_on = False

dataset = input_pipeline.parse()
iterator = dataset.make_initializable_iterator()

ckpt_folder = './ckpt/working_copy'

####### get input #######
input_image, resulting_image = iterator.get_next()

####### make prediction #######
latent_space, prediction = network.encoder_decoder(input_image)

if disc_on == True:
	fake_prediction = network.discriminator(prediction)
	real_prediction = network.discriminator(input_image,True)
	loss_generator = losses_helper.generator_loss(fake_prediction)
	loss_discriminator = losses_helper.discriminator_loss(real_prediction,fake_prediction)

	d_opt = tf.train.AdamOptimizer(learning_rate,beta1=0.5, beta2=0.999).minimize(total_disc_loss)


loss_recon = losses_helper.reconstruction_loss_l1(prediction,input_image)
loss_kl = losses_helper.KL_divergence_loss(latent_space)



total_vae_loss = loss_recon * recon_loss_weight + loss_kl * kl_loss_weight 


if disc_on == True:
	total_vae_loss +=  loss_generator * GAN_weight
	total_disc_loss = loss_discriminator * GAN_weight


MAX_ITERATIONS = 10000
alternate_global_step = tf.placeholder(tf.int32)

global_step = tf.get_variable(
    'global_step', [],
    initializer=tf.constant_initializer(0), trainable=False)

learning_rate = tf.train.polynomial_decay(0.0001, alternate_global_step,
                                          MAX_ITERATIONS, 0.000001,
                                          power=3)

g_opt = tf.train.AdamOptimizer(learning_rate,beta1=0.5, beta2=0.999).minimize(total_vae_loss)
 

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
train_summaries.append(tf.summary.image('prediction',prediction))

if disc_on == True:
	train_summaries.append(tf.summary.scalar('loss_generator',loss_generator))
	train_summaries.append(tf.summary.scalar('total_disc_loss',total_disc_loss))
# train_summaries.append(tf ummary.image('predicted_image',prediction))

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

folder_name = ckpt_folder.split('/')[-1]
for step in range(loop_start,loop_stop+1):


	# generator_iterations = 1

	# for i in range(generator_iterations):
	_ , g_loss = sess.run([g_opt,total_vae_loss],feed_dict={
			alternate_global_step: iteration
	})

	format_str = ('%s: step %d, g_loss = %.15f, folder = %s')
	print((format_str % (datetime.now(),step, g_loss, folder_name)))


	if disc_on == True:
		_ , d_loss = sess.run([d_opt,total_disc_loss],feed_dict={
				alternate_global_step: iteration
		})
		format_str = ('%s: step %d, d_loss = %.15f, folder = %s')
		print((format_str % (datetime.now(),step, d_loss, folder_name)))




	# assert not np.isnan(loss), 'Model diverged with loss = NaN'



	print('')

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






