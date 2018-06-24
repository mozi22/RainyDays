
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

####### get input #######
input_image, resulting_image = iterator.get_next()



####### make prediction #######
prediction, z_mu, z_sigma = network.create_network(input_image)

####### define losses #######

# resize input_image for calculating reconstruction loss on a slightly smaller image.
input_image_resized = tf.image.resize_images(input_image,[224,224],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

loss_recon = losses_helper.reconstruction_loss(prediction,input_image_resized)
loss_kl = losses_helper.KL_divergence_loss(z_mu,z_sigma)
total_loss = tf.reduce_mean(loss_recon + loss_kl)



####### initialize optimizer #######

MAX_ITERATIONS = 50000

global_step = tf.get_variable(
    'global_step', [],
    initializer=tf.constant_initializer(0), trainable=False)

learning_rate = tf.train.polynomial_decay(0.0001, global_step,
                                          MAX_ITERATIONS, 0.000001,
                                          power=3)

opt = tf.train.AdamOptimizer(learning_rate)
grads = opt.compute_gradients(total_loss)
apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

# Track the moving averages of all trainable variables.
variable_averages = tf.train.ExponentialMovingAverage(
    0.9999, global_step)
variables_averages_op = variable_averages.apply(tf.trainable_variables())

# Group all updates to into a single train op.
train_op = tf.group(apply_gradient_op, variables_averages_op)



####### write summaries #######

train_summaries = []

train_summaries.append(tf.summary.scalar('recon_loss',loss_recon))
train_summaries.append(tf.summary.histogram('prediction',prediction))
train_summaries.append(tf.summary.histogram('gt',input_image_resized))
train_summaries.append(tf.summary.scalar('kl_loss',loss_kl))
train_summaries.append(tf.summary.scalar('total_loss',total_loss))
train_summaries.append(tf.summary.image('input_image',input_image))
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


summary_writer = tf.summary.FileWriter('./ckpt', sess.graph)


first_iteration = True

for step in range(0,MAX_ITERATIONS):


	_ , loss = sess.run([train_op,total_loss])


	assert not np.isnan(loss), 'Model diverged with loss = NaN'


	format_str = ('%s: step %d, loss = %.15f')
	print((format_str % (datetime.now(),step, loss)))

	if step % 100 == 0:
		summmary = sess.run(summary_op)
		summary_writer.add_summary(summmary,step)

	# Save the model checkpoint periodically.
	if step % 500 == 0:
		checkpoint_path = os.path.join('./ckpt', 'model.ckpt')
		saver.save(sess, checkpoint_path, global_step=step)


summary_writer.close()