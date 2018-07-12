
import network
import input_pipeline
import tensorflow as tf
import numpy as np
import time
import losses_helper
import os
from datetime import datetime


L1_weight = 100
L1_cycle_weight = 100
KL_weight = 0.1
KL_cycle_weight = 0.1
GAN_weight = 10


disc_on = False

dataset = input_pipeline.parse()
iterator = dataset.make_initializable_iterator()

ckpt_folder = './ckpt/working_copy'

####### get input #######
imageA, imageB = iterator.get_next()

####### make prediction #######
input_Aa, input_Ab, input_Ba, input_Bb, enc_shared_latent_space = network.perform_image_translation(imageA, imageB)
result_Ab, sharedAb = network.generate_atob(input_Ab)
result_Ba, sharedBa = network.generate_btoa(input_Ba)



real_A_logit, real_B_logit = network.discriminate_real(imageA, imageB)
fake_A_logit, fake_B_logit = network.discriminate_fake(input_Ab, input_Ba)


G_ad_loss_a = losses_helper.generator_loss(fake_A_logit)
G_ad_loss_b = losses_helper.generator_loss(fake_B_logit)

D_ad_loss_a = losses_helper.discriminator_loss(real_A_logit, fake_A_logit)
D_ad_loss_b = losses_helper.discriminator_loss(real_B_logit, fake_B_logit)

loss_reconAb = losses_helper.reconstruction_loss_l1(result_Ab,imageB)
loss_reconBa = losses_helper.reconstruction_loss_l1(result_Ba,imageA)
loss_reconB = losses_helper.reconstruction_loss_l1(input_Bb,imageB)
loss_reconA = losses_helper.reconstruction_loss_l1(input_Aa,imageA)

loss_kl_shared = losses_helper.KL_divergence_loss(enc_shared_latent_space)
loss_kl_Ab = losses_helper.KL_divergence_loss(sharedAb)
loss_kl_Ba = losses_helper.KL_divergence_loss(sharedBa)


Generator_A_loss = GAN_weight * G_ad_loss_a + \
                   L1_weight * loss_reconA + \
                   L1_cycle_weight * loss_reconBa + \
                   KL_weight * loss_kl_shared + \
                   KL_cycle_weight * loss_kl_Ab

Generator_B_loss = GAN_weight * G_ad_loss_b + \
                   L1_weight * loss_reconB + \
                   L1_cycle_weight * loss_reconAb + \
                   KL_weight * loss_kl_shared + \
                   KL_cycle_weight * loss_kl_Ba

Generator_loss = Generator_A_loss + Generator_B_loss

Discriminator_A_loss = GAN_weight * D_ad_loss_a
Discriminator_B_loss = GAN_weight * D_ad_loss_b

Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss

MAX_ITERATIONS = 20000

alternate_global_step = tf.placeholder(tf.int32)

global_step = tf.get_variable(
    'global_step', [],
    initializer=tf.constant_initializer(0), trainable=False)

learning_rate = tf.train.polynomial_decay(0.0001, alternate_global_step,
                                          MAX_ITERATIONS, 0.000001,
                                          power=3)

t_vars = tf.trainable_variables()
G_vars = [var for var in t_vars if ('generator' in var.name) or ('encoder' in var.name) or ('shared_latent_space' in var.name)]
D_vars = [var for var in t_vars if 'discriminator' in var.name]



g_opt = tf.train.AdamOptimizer(learning_rate,beta1=0.5, beta2=0.999).minimize(Generator_loss, var_list=G_vars)
d_opt = tf.train.AdamOptimizer(learning_rate,beta1=0.5, beta2=0.999).minimize(Discriminator_loss, var_list=D_vars)
 

####### write summaries #######

train_summaries = []

train_summaries.append(tf.summary.image('AtoB',result_Ab))
train_summaries.append(tf.summary.image('BtoA',result_Ba))
train_summaries.append(tf.summary.image('AtoA',input_Aa))
train_summaries.append(tf.summary.image('BtoB',input_Bb))

train_summaries.append(tf.summary.scalar('gen_loss',Generator_loss))
train_summaries.append(tf.summary.scalar('dis_loss',Discriminator_loss))
train_summaries.append(tf.summary.scalar('KL_trans',loss_kl_shared))
train_summaries.append(tf.summary.scalar('loss_kl_Ab',loss_kl_Ab))
train_summaries.append(tf.summary.scalar('loss_kl_Ba',loss_kl_Ba))

train_summaries.append(tf.summary.scalar('recon_AA',loss_reconA))
train_summaries.append(tf.summary.scalar('recon_AB',loss_reconAb))
train_summaries.append(tf.summary.scalar('recon_BA',loss_reconBa))
train_summaries.append(tf.summary.scalar('recon_BB',loss_reconB))



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
	_ , g_loss = sess.run([g_opt,Generator_loss],feed_dict={
			alternate_global_step: iteration
	})

	format_str = ('%s: step %d, g_loss = %.15f, folder = %s')
	print((format_str % (datetime.now(),step, g_loss, folder_name)))


	_ , d_loss = sess.run([d_opt,Discriminator_loss],feed_dict={
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






