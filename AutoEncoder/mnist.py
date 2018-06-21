import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def load_mnist():
    """
    :return: train_data, valid_data, test_data

    Data type:
    tuple (raw_data, one_hot_label)
    """

    from tensorflow.examples.tutorials.mnist import input_data

    # Load MNIST Data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    train_data = mnist.train
    valid_data = mnist.validation
    test_data = mnist.test

    input_shape, label_shape = train_data.next_batch(1)[0].shape[1], train_data.next_batch(1)[1].shape[1]
    return train_data, valid_data, test_data, input_shape, label_shape

class AutoEncoder(object):
    def __init__(self, n_features, layers=[], code_nodes=2, learning_rate= 0.013):
        self.code_nodes = code_nodes

        self.graph = tf.Graph() # initialize new graph
        self.model(n_features, learning_rate, layers)
        self.sess = tf.Session(graph=self.graph) # initialize session

    def model(self, n_features, learning_rate, layers):
        """
        Build model with tensorflow
        :return: model
        """
        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, shape=(None, n_features), name="X_Input")

            layer_input = self.X

            # Encoder
            for nodes in layers:
                layer_input = tf.layers.dense(layer_input,
                                             nodes,
                                             activation=tf.nn.relu,
                                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                             name="FC_Encode_Layer_" + str(nodes))

            # encode layer
            self.code = tf.layers.dense(layer_input,
                                  encode_nodes,
                                  activation=tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                  name="Code")

            layer_input = self.code

            # Decoder
            for nodes in reversed(layer_nodes):
                layer_input = tf.layers.dense(layer_input,
                                             nodes,
                                             activation=tf.nn.relu,
                                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                             name="FC_Decode_Layer_" + str(nodes))

            self.y_ = tf.layers.dense(layer_input,
                                  784,
                                  activation=tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                  name="y_pred")

            self.loss = tf.reduce_mean(tf.pow(self.y_ - self.X, 2))  # mean square error
            self.optimizer = tf.train.AdamOptimizer(learning_rate)

            self.train_op = self.optimizer.minimize(self.loss) # learning goal

            self.init_op = tf.global_variables_initializer() # initialization

    def fit(self, X, X_valid, X_test=None, epochs = 10, batch_size=32, total_sample=1000):
        self.sess.run(self.init_op) # initializing

        for epoch in range(epochs):
            print("Epoch #", epoch+1)

            rounds = int(total_sample/batch_size)
            for i in range(rounds):
                X_batch = X.next_batch(batch_size)
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.X: X_batch[0]})
                if i % 250 == 0:
                    print("Train [%d/%d] loss = %9.4f" % (i+1, rounds, loss))

            # validation
            [loss] = self.sess.run([self.loss], feed_dict={self.X: X_valid.images})
            print("Valid loss = %9.4f" % loss)

            # testing
            if X_test:
                [loss] = self.sess.run([self.loss], feed_dict={self.X: X_test.images})
                print("Test loss = %9.4f" % loss)


    def predict(self, X):
        code, y_pred = self.sess.run([self.code, self.y_], feed_dict={self.X: X})
        return code, y_pred


if __name__ == '__main__':
    train_data, valid_data, test_data, input_shape, label_shape = load_mnist()

    layer_nodes = [512, 256, 128, 64, 32]
    # layer_nodes = [515, 512, 509, 506, 503]
    encode_nodes = 16

    auto_encoder = AutoEncoder(input_shape, layer_nodes, encode_nodes, learning_rate=0.00013)

    auto_encoder.fit(train_data, valid_data, test_data, total_sample=train_data.images.shape[0], epochs= 15)

    for i in range(5):
        img = np.reshape(test_data.images[i, :], (28, 28)) # re-shape to image like

        c, y_pred = auto_encoder.predict(np.reshape(test_data.images[i, :], (1, 28*28)))
        img_pred = np.reshape(y_pred, (28, 28)) # re-shape to image like

        plt.matshow(img, cmap=plt.get_cmap('gray'))
        plt.matshow(img_pred, cmap=plt.get_cmap('gray'))

    plt.show()
    plt.close()