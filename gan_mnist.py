import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from plotter import Plotter
import time

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

def xavier_init(shape):
    return tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)(shape)


# Discriminator net
X = tf.placeholder(tf.float32, shape=[None, 784], name='X')

D_w1 = tf.Variable(xavier_init([784, 128]), name='D_w1')
D_b1 = tf.Variable(tf.zeros(shape=[128]), name='D_b1')

D_w2 = tf.Variable(xavier_init([128,1]), name='D_w2')
D_b2 = tf.Variable(tf.zeros(shape=[1]), name='D_b2')

# Generator net
Z_dim = 100
Z = tf.placeholder(tf.float32, shape=[None, Z_dim], name='Z')

G_w1 = tf.Variable(xavier_init([Z_dim, 128]), name='G_w1')
G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')

G_w2 = tf.Variable(xavier_init([128,784]), name='G_w2')
G_b2 = tf.Variable(tf.zeros(shape=[784]), name='G_b2')

theta_D = [D_w1, D_w2, D_b1, D_b2]
theta_G = [G_w1, G_w2, G_b1, G_b2]

def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_w1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_w2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob

def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_w1) + D_b1)
    D_logit = tf.matmul(D_h1, D_w2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit

G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)
D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))

#Solvers
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m,n])

saver = tf.train.Saver()
with tf.Session() as sess:
    chkpt_path = tf.train.latest_checkpoint('./checkpoints')
    if chkpt_path:
        saver.restore(sess, chkpt_path)
    else:
        init = tf.global_variables_initializer()
        sess.run(init)

    #Start algorithm
    n_iterations = 4000
    k = 1
    i = 0
    mb_size = 16
    plotter = Plotter()
    for iteration in range(n_iterations):
        for step in range(k):
            X_mb, _ = mnist.train.next_batch(mb_size)
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_z(mb_size, Z_dim)})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_z(mb_size, Z_dim)})

        if iteration % 200 == 0:
            samples = sess.run(G_sample, feed_dict={Z: sample_z(mb_size, Z_dim)})
            plotter.plot(samples)
            # plt.savefig('out/{}.png'
            #             .format(str(i).zfill(3)), bbox_inches='tight')
            # i += 1
            print(iteration)
            saver.save(sess, './checkpoints/prev_sess')
