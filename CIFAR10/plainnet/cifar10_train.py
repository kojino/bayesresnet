# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.
Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.
Speed: With batch_size 128.
System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.models.image.cifar10 import cifar10

# app is a simple wrapper that handles flag
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_plainnet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_float('keep_prob', 0.5,
                            """Dropout rate""")

def train():
  """Train CIFAR-10 for a number of steps."""
  # A TensorFlow computation, represented as a dataflow graph.
  g = tf.Graph()
  # Overrides the current default graph for the lifetime of the context
  with g.as_default():
    # Count the number of training steps processed.
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for CIFAR-10.
    images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    # Create a saver to save all variables in the model
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation that merges all summaries collected in the default graph.
    summary_op = tf.summary.merge_all()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Create a session.
    # ConfigProto is simply telling the session to log the placement decisions
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))

    # Run initialize variable operation.
    sess.run(init)

    # Start the queue runners. Use sess session to run all the queues collected.
    tf.train.start_queue_runners(sess=sess)

    # On construction of FileWriter, a new event file is created in logdir.
    # This event file contains Event protocol buffers for adding logs in the future.
    # Pass graph from the session launched
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                            graph=sess.graph)

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      # Run train and loss calculation operations.
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      # Make sure that the loss is not nan
      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      ##### Operations for summary recording below #####

      # Every 10 steps, show step, loss, examples/sec and sec/batch.
      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      # Every 100 steps, create a summary and add it to log.
      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Every 1000 steos (or in the last step), save the model checkpoint.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  # download cifar10 if not downloaded
  cifar10.maybe_download_and_extract()
  # delete existing logs
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  # recreate the directory for logs
  tf.gfile.MakeDirs(FLAGS.train_dir)
  # start training
  train()


if __name__ == '__main__':
  # ensures that any flags are parsed,
  # and then invokes the main() function in the same module
  tf.app.run()
