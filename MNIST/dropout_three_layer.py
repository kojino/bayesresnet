
# coding: utf-8

# In[3]:

import tensorflow as tf
import numpy
from tensorflow.examples.tutorials.mnist import input_data


# In[4]:

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_iter', 20000, "Number of training iterations")
tf.app.flags.DEFINE_integer('batch_size', 50, "Batch size per iteration")


# In[ ]:

class DropNet():
    def __init__(self, hps, images, labels, mode):
        """ResNet constructor.
        Args:
          hps: Hyperparameters.
          images: Batches of images. [batch_size, image_size, image_size, 3]
          labels: Batches of labels. [batch_size, num_classes]
          mode: One of 'train' and 'eval'.
        """
        self._images = images
        self.labels = labels
        self.mode = mode
        self.hps = hps
        
    def build_model(self):
        """Build the core model within the graph."""
        with tf.variable_scope('init'):
            x = self._images

        with tf.variable_scope('fc'):
            z = self.fully_connected(x, self.hps.kernel_shape, self.hps.bias_shape)
        
        with tf.variable_scope('logit'):
            logits = self.logit(z, self.hps.num_classes)
            self.predictions = tf.nn.softmax(logits)

        with tf.variable_scope('costs'):
            xent = tf.nn.softmax_cross_entropy_with_logits(
              logits=logits, labels=self.labels)
            self.cost = tf.reduce_mean(xent, name='xent')
            self.cost += self._decay()

        tf.summary.scalar('cost', self.cost)
        
    def fully_connected(self, input, kernel_shape, bias_shape):
        '''
        one fully connected layer 
        input: input for the layer, kernel shape and bias shape
        output: z of the layer, tf graph is constructed
        '''
        # Create variable named "weights".
        weights = tf.get_variable("weights", kernel_shape,
            initializer=tf.random_normal_initializer())
        # Create variable named "biases".
        biases = tf.get_variable("biases", bias_shape,
            initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input, weights,
            strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.relu(tf.matmul(input, weights) + biases)

        
    def logit(self, x, out_dim):
        """FullyConnected layer for final output."""
        x = tf.reshape(x, [self.hps.batch_size, -1])
        w = tf.get_variable(
            'DW', [x.get_shape()[1], out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)


# In[7]:

def build_training_graph():
    '''
    1.
    creates x, y placeholder
    trains the graph with forward prop followed by backward prop
    '''
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    logit = forward(x) # 1.1
    loss = backward(logit, y) #1.2


# In[ ]:

def forward(x, is_traning=True):
    '''
    1.1, 2.1
    forward prop for training the neural network
    NOW: Three layer fc NN.
    '''
    with tf.variable_scope("fc1"):
        # Variables created here will be named "fc1/weights", "fc1/biases".
        relu1 = fc_layer(x, [5, 5, 32, 32], [32])
    with tf.variable_scope("fc2"):
        # Variables created here will be named "fc2/weights", "fc2/biases".
        return fc_layer(relu1, [5, 5, 32, 32], [32])


# In[ ]:

def backward(logit,y):
    '''
    1.2, 2.2
    NOW: minimize softmax cross entropy
    '''
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logit, y))
    return loss


# In[ ]:

def build_eval_graph():
    '''
    2.
    creates x, y placeholder
    evaluates the graph with forward prop followed by backward prop
    '''
    x = tf.placeholder(tf.float32, [None, layer_sizes[0]])
    y_ = tf.placeholder(tf.float32, [None, layer_sizes[-1]])
    y_pred = forward(x, is_training=False)
    correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_,1))
    loss = backward(logit,y)
    accuracy = 


# In[6]:

def fc_layer(input, kernel_shape, bias_shape):
    '''
    one fully connected layer 
    input: input for the layer, kernel shape and bias shape
    output: z of the layer, tf graph is constructed
    '''
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(tf.matmul(input, weights) + biases)


# In[ ]:

def main(_):
    with tf.Graph().as_default():
        with tf.variable_scope("MNIST_NN") as scope:
            build_training_graph() #1
            scope.reuse_variables()
            build_eval_graph() #2
            
        init = tf.initialize_all_variables()
        sess = tf.Session()
        
    sess.run(init)
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    for i in range(FLAGS.num_iter):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
              x:batch[0], y_: batch[1], is_train: False})
            print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], is_train: True})


# In[ ]:

if __name__ == "__main__":
    tf.app.run()

