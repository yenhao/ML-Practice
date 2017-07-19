import tensorflow as tf

"""
Load the mnist dataset from tensorflow
"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

"""
Define Code

Set the input nodes

Not understand:
Represent this as a 2-D tensor of floating-point numbers, with a shape [None, 784].
(Here None means that a dimension can be of any length.)
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
"""
y_ = tf.placeholder(tf.float32, [None, 10])

"""
Define Code

Cross entropy formula as lost function
"""
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


"""
Define Code

Define the optimizer and the best critia (minimize the cross_entropy function)
"""
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

"""
launch model

Not understand:
Difference between Session & InteractiveSession
"""
sess = tf.InteractiveSession()

"""
Execute Code

Initialize the variables
"""
tf.global_variables_initializer().run()

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

"""
Get about 92% accuracy in document

But I only get 90%
"""
