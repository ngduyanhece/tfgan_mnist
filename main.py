# Make TFGAN models and TF-Slim models discoverable.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import functools
import tensorflow as tf
tfgan = tf.contrib.gan
import data_provider
import util
from datasets import download_and_convert_mnist
from mnist_train_util import infogan_generator
from mnist_train_util import infogan_discriminator
from mnist_train_util import  save_image

slim = tf.contrib.slim
layers = tf.contrib.layers
ds = tf.contrib.distributions
batch_size = 32
# noise_dims = 64
cat_dim, cont_dim, noise_dims = 10, 2, 64
MNIST_DATA_DIR = './mnist-data'
MNIST_IMAGE_DIR = './mnist-image'
if __name__ == '__main__':
    if not tf.gfile.Exists(MNIST_DATA_DIR):
        tf.gfile.MakeDirs(MNIST_DATA_DIR)
        download_and_convert_mnist.run(MNIST_DATA_DIR)
    if not tf.gfile.Exists(MNIST_IMAGE_DIR):
        tf.gfile.MakeDirs(MNIST_IMAGE_DIR)
    images, one_hot_labels, _ = data_provider.provide_data('train', batch_size, MNIST_DATA_DIR)

    generator_fn = functools.partial(infogan_generator, categorical_dim=cat_dim)
    discriminator_fn = functools.partial(
        infogan_discriminator, categorical_dim=cat_dim,
        continuous_dim=cont_dim)
    unstructured_inputs, structured_inputs = util.get_infogan_noise(
        batch_size, cat_dim, cont_dim, noise_dims)

    infogan_model = tfgan.infogan_model(
        generator_fn=generator_fn,
        discriminator_fn=discriminator_fn,
        real_data=images,
        unstructured_generator_inputs=unstructured_inputs,
        structured_generator_inputs=structured_inputs)
    infogan_loss = tfgan.gan_loss(
        infogan_model,
        gradient_penalty_weight=1.0,
        mutual_information_penalty_weight=1.0)
    generator_optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
    discriminator_optimizer = tf.train.AdamOptimizer(0.00009, beta1=0.5)
    gan_train_ops = tfgan.gan_train_ops(
        infogan_model,
        infogan_loss,
        generator_optimizer,
        discriminator_optimizer)

    train_step_fn = tfgan.get_sequential_train_steps()
    global_step = tf.train.get_or_create_global_step()
    loss_values, mnist_score_values = [], []
    generated_data_to_visualize = tfgan.eval.image_reshaper(
        infogan_model.generated_data[:20, ...], num_cols=10)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with slim.queues.QueueRunners(sess):
            start_time = time.time()
            for i in range(6001):
                cur_loss, _ = train_step_fn(
                    sess, gan_train_ops, global_step, train_step_kwargs={})
                loss_values.append((i, cur_loss))
                if i % 200 == 0:
                    print('Current epoch: %d' % i)
                    print('Current loss: %f' % cur_loss)
                    img_name = "result_" + str(i)
                    save_image(sess.run(generated_data_to_visualize),MNIST_IMAGE_DIR,img_name)