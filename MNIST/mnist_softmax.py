
# coding: utf-8

# In[1]:

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# In[2]:

import tensorflow as tf
sess = tf.InteractiveSession()


# In[3]:

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# In[4]:

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


# In[5]:

sess.run(tf.global_variables_initializer())


# In[6]:

y = tf.matmul(x,W) + b


# In[7]:

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))


# In[8]:

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# In[9]:

for i in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})


# In[10]:

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))


# In[11]:

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[12]:

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# In[ ]:



