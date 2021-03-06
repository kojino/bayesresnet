
# coding: utf-8

# In[1]:

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# In[2]:

import tensorflow as tf
import numpy as np
sess = tf.InteractiveSession()


# In[3]:

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# In[4]:

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# In[5]:

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# In[6]:

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])


# In[7]:

x_image = tf.reshape(x, [-1,28,28,1])


# In[8]:

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# In[9]:

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# In[10]:

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# In[11]:

is_train = tf.placeholder(tf.bool)

def create_dropout_matrix():
    means = tf.fill([1,1024], 0.5)
    a = tf.select(tf.random_uniform([1, 1024])- means > 0, tf.ones([1,1024]), tf.zeros([1,1024]))
    return tf.diag(a[0])

def if_train():
    return create_dropout_matrix()

def if_not_train():
    I_drop = create_dropout_matrix()
    return tf.mul(I_drop,2.0)

I_drop = tf.cond(is_train, if_train, if_not_train)


# In[12]:

W_fc2 = weight_variable([1024, 10])
W_fc2_drop = tf.matmul(I_drop, W_fc2)
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1, W_fc2_drop) + b_fc2


# In[15]:

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())


# In[ ]:

keep_prob = 0.5

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
          x:batch[0], y_: batch[1], is_train: False})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], is_train: True})


# In[17]:

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, is_train: False}))


# In[ ]:



