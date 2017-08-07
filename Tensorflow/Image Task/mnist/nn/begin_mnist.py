import tensorflow as tf

"""
Load the mnist dataset from tensorflow
"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

"""
Define Code

Set the input nodes

Represent this as a 2-D tensor of floating-point numbers, with a shape [None, 784].
(Here None means that a dimension can be of any length.)

The input images x will consist of a 2d tensor of floating point numbers.
Here we assign it a shape of [None, 784], where 784 is the dimensionality of a single flattened 28 by 28 pixel MNIST image,
 and None indicates that the first dimension, corresponding to the batch size, can be of any size. 
"



"""
x = tf.placeholder(tf.float32, [None, 784])

"""
Define Code

Set Weight & bios

10 also means that it got ten nodes from next layer
"""
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

"""
Define Code

Apply softmax function after the propagation

Softmax function is able to let output array's value normalized and also make them sum to 1.
It's kinds of probability representation.
"""
y = tf.nn.softmax(tf.matmul(x, W) + b)

"""
Define Code

Set the correct answer (label) input nodes
The target output classes y_ will also consist of a 2d tensor,
 where each row is a one-hot 10-dimensional vector indicating which digit class (zero through nine) the corresponding MNIST image belongs to.
"""
y_ = tf.placeholder(tf.float32, [None, 10])

"""
Define Code

Cross entropy formula as lost function

Not understand:
reduction_indices=[1]
"""
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# cross_entropy = tf.reduce_mean(
    # tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

"""
Define Code

Define the optimizer and the best critia (minimize the cross_entropy function)
"""
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

"""
launch model

Not understand:
Difference between Session & InteractiveSession

Explaination

InteractiveSession class, which makes TensorFlow more flexible about how you structure your code. It allows you to interleave operations which build a computation graph with ones that run the graph. This is particularly convenient when working in interactive contexts like IPython. If you are not using an InteractiveSession, then you should build the entire computation graph before starting a session and launching the graph.

"""
sess = tf.InteractiveSession()

"""
Execute Code

Initialize the variables

Before Variables can be used within a session, they must be initialized using that session. This step takes the initial values (in this case tensors full of zeros) that have already been specified, and assigns them to each Variable. This can be done for all Variables at once
"""
tf.global_variables_initializer().run()

# sess.run(tf.global_variables_initializer()) # another usage

"""
Execute Code

Train for 1000 rounds
"""
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100) # random take 100 from datasets to train
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) # Feed the data & Train

"""
Evaluation
"""

"""
Define Code

Take the index of y/y_ that is most closest to 1
If two are equal, return True, other False
"""
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

"""
Define Code

Boolean to 0/1, then average it

ex: [True, False, True, True] would become [1,0,1,1] which would become 0.75
"""
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""
Execute Code

To get the accuracy
"""
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})) # function same as previous one

"""
Get about 92% accuracy in document

But I only get 90% in second cross entropy
And get ~92% in first cross entropy
"""
