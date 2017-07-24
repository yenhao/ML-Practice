import tensorflow as tf

# Data Input
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./tmp/data/", one_hot=True)

# Network variables
learning_rate = 0.001
training_iters = 20000
batch_size = 128
display_step = 10

n_input = 784 #( 28 * 28)
n_classes = 10 #( 0 ~ 9)
dropout = 0.75 

"""
Define Graph
"""
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout


def conv2d(name, x, W, b, strides=1):
    """
    Define Convention
    """
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x, name=name) # Activative function

def maxpool2d(name, x, k=2):
    """
    Define Pooling Layer
    """
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

def norm(name, l_input, lsize=4):
    """
    Define Normalization
    """
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001/9.0, beta=0.75, name=name)


"""
Parameters
"""
weights = {
    'wc1' : tf.Variable(tf.random_normal([11, 11, 1, 96])),
    'wc2' : tf.Variable(tf.random_normal([5, 5, 96, 256])),
    'wc3' : tf.Variable(tf.random_normal([3, 3, 256, 384])),
    'wc4' : tf.Variable(tf.random_normal([3, 3, 384, 384])),
    'wc5' : tf.Variable(tf.random_normal([3, 3, 384, 256])),
    'wd1' : tf.Variable(tf.random_normal([4*4*256, 4096])),
    'wd2' : tf.Variable(tf.random_normal([4096, 4096])),
    'out' : tf.Variable(tf.random_normal([4096, n_classes]))
}

biases = {
    'bc1' : tf.Variable(tf.random_normal([96])),
    'bc2' : tf.Variable(tf.random_normal([256])),
    'bc3' : tf.Variable(tf.random_normal([384])),
    'bc4' : tf.Variable(tf.random_normal([384])),
    'bc5' : tf.Variable(tf.random_normal([256])),
    'bd1' : tf.Variable(tf.random_normal([4096])),
    'bd2' : tf.Variable(tf.random_normal([4096])),
    'out' : tf.Variable(tf.random_normal([n_classes]))
}


def alex_net(x, weights, biases, dropout):
    """
    Define AlexNet Network Structure
    """
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # First convention layer
    conv1 = conv2d('conv1', x, weights['wc1'], biases['bc1'])
    # Pooling
    pool1 = maxpool2d('pool1', conv1, k=2)
    # Normalization
    norm1 = norm('norm1', pool1, lsize=4)

    # Second convention layer
    conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
    # Pooling
    pool2 = maxpool2d('pool2', conv2, k=2)
    # Normalization
    norm2 = norm('norm2', pool2, lsize=4)

    # Third convention layer
    conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
    # Pooling
    # pool3 = maxpool2d('pool3', conv3, k=2)
    # Normalization
    norm3 = norm('norm3', conv3, lsize=4)

    # Fourth convention layer
    conv4 = conv2d('conv4', norm3, weights['wc4'], biases['bc4'])
    # Fifth convention layer
    conv5 = conv2d('conv5', conv4, weights['wc5'], biases['bc5'])
    # Pooling
    pool5 = maxpool2d('pool5', conv5, k=2)
    # Normalization
    norm5 = norm('norm5', pool5, lsize=4)


    # Fully Connective layer 1
    fc1 = tf.reshape(norm5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Fully Connective layer 2
    fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    # dropout
    fc2 = tf.nn.dropout(fc2, dropout)

    # output layer
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out

"""
Build Model
"""
pred = alex_net(x, weights, biases, keep_prob)

"""
Lost function + Optimizer
"""
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

"""
Evaluation
"""
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


"""
Execute Code
"""

# Init
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Training Begin, until training_iters (200000) 
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

        if step % display_step == 0:
        # Out put lost & accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})

            print("Iter {}, Minibatch Loss = {:.6f}, Training Accuracy= {:.5f}".format(step*batch_size, loss, acc))

        step += 1

    print("Optimization Finished!")

    #Test the accuracy by testing data
    print("Testing Accuracy:",\
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.})
        )


"""
Got Error..

InvalidArgumentError (see above for traceback): logits and labels must be same size: logits_size=[32,10] labels_size=[128,10]
"""