
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

class dropout(x, keep_prob):
    dist = Binomial(n=1024, p=keep_prob)
    drop_x = 
    


# In[7]:

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])


# In[8]:

x_image = tf.reshape(x, [-1,28,28,1])


# In[9]:

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# In[10]:

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# In[11]:

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# In[12]:

print(h_fc1.get_shape())


# In[13]:

# idx = tf.tile(a, [3, 1])
# with tf.Session(''): print(idx.eval(),idx.get_shape())

# idx = tf.reshape(a, [-1, 1])    # Convert to a len(yp) x 1 matrix.
# idx = tf.tile(idx, [1, 3])
# with tf.Session(''): print(idx.eval(),idx.get_shape())
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# idx = tf.tile(a, [10, 1])
# with tf.Session(''): print(a.eval(),a.get_shape())


# In[14]:

means = tf.fill([1,1024], 0.5)
a = tf.select(tf.random_uniform([1, 1024])- means > 0, tf.ones([1,1024]), tf.zeros([1,1024]))


# In[15]:

I_drop = tf.placeholder(tf.float32, shape=(1024, 1024))


# In[17]:

W_fc2 = weight_variable([1024, 10])
W_fc2_drop = tf.matmul(I_drop, W_fc2)
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1, W_fc2_drop) + b_fc2


# In[18]:

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())


# In[26]:

for i in range(20000):
    batch = mnist.train.next_batch(50)
    
    # prepare identity matrix
    identity = np.identity(1024)
    
    # prepare dropout matrix
    matrix = np.identity(1024)
    s = np.random.binomial(1, 0.5, 1024)
    for i,el in enumerate(s):
        if el == 0:
        matrix[i][i] = 0
        
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
          x:batch[0], y_: batch[1], I_drop: identity})
    print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], I_drop: matrix})


# In[15]:

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


# In[ ]:



