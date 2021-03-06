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
    def __init__(self, n_features=0, layers=[], code_nodes=2, learning_rate= 0.013, alpha=0.1, scale=0.073, load=None):
        self.code_nodes = code_nodes

        self.regularizer = tf.contrib.layers.l2_regularizer(scale=scale)
        self.alpha = alpha

        self.graph = tf.Graph() # initialize new graph
        if load: # load pre-trained
            self.load_model(load)
            self.sess = tf.Session(graph=self.graph)  # initialize session
            self.restore_saver.restore(self.sess, tf.train.latest_checkpoint('/'.join(load.split('/')[:-1])))
            print(tf.train.latest_checkpoint('/'.join(load.split('/')[:-1])))
            print('Model loaded!')

        else: # train a new model
            self.model(n_features, learning_rate, layers)
            self.sess = tf.Session(graph=self.graph) # initialize session



    def load_model(self, path):
        with self.graph.as_default():
            self.restore_saver = tf.train.import_meta_graph(path)  # load meta graph

        print(self.graph.get_operations())
        self.X = self.graph.get_tensor_by_name('X_Input:0')

        self.code = self.graph.get_tensor_by_name('Code/Relu:0').op.outputs[0]

        self.loss = self.graph.get_tensor_by_name('add_1:0').op.outputs[0]


        self.y_ = self.graph.get_tensor_by_name('y_pred/Sigmoid:0').op.outputs[0]


    def model(self, n_features, learning_rate, layers):
        """
        Build model with tensorflow
        :return: Nothing
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
                                             kernel_regularizer= self.regularizer,
                                             name="FC_Encode_Layer_" + str(nodes))

            # encode layer
            self.code = tf.layers.dense(layer_input,
                                  encode_nodes,
                                  activation=tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                  kernel_regularizer= self.regularizer,
                                  name="Code")

            layer_input = self.code

            # self.code = layer_nodes


            # Decoder
            for nodes in reversed(layer_nodes):
                layer_input = tf.layers.dense(layer_input,
                                             nodes,
                                             activation=tf.nn.relu,
                                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                             kernel_regularizer= self.regularizer,
                                             name="FC_Decode_Layer_" + str(nodes))

            self.y_ = tf.layers.dense(layer_input,
                                  784,
                                  activation=tf.nn.sigmoid,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                  kernel_regularizer=self.regularizer,
                                  name="y_pred")

            self.mse_loss = tf.reduce_mean(tf.pow(self.y_ - self.X, 2)) # mean square error
            self.l2_loss = tf.losses.get_regularization_loss() # l2 loss

            self.loss = self.mse_loss + self.alpha * self.l2_loss/(1+self.l2_loss) # final loss

            tf.summary.scalar('mse_loss', self.mse_loss)
            tf.summary.scalar('l2_loss', self.l2_loss)
            tf.summary.scalar('total_loss', self.loss)

            self.optimizer = tf.train.AdamOptimizer(learning_rate)

            self.train_op = self.optimizer.minimize(self.mse_loss) # learning goal

            self.init_op = tf.global_variables_initializer() # initialization

            self.merged = tf.summary.merge_all()

            self.saver = tf.train.Saver()  # to store the model

    def fit(self, X, X_valid, X_test=None, epochs = 10, batch_size=32, total_sample=1000):
        """
        Starting to training our Auto Encoder

        :param X: training sample [np array]
        :param X_valid: validation sample [np array]
        :param X_test: test sample [np array]
        :param epochs: nums of epochs
        :param batch_size: nums of batch size
        :param total_sample: numbers of sample for training samples
        :return: nothing
        """

        self.writer = tf.summary.FileWriter("TensorBoard/", graph=self.graph)

        self.sess.run(self.init_op) # initializing

        for epoch in range(epochs):
            print("Epoch #", epoch+1)

            rounds = int(total_sample/batch_size)
            for i in range(rounds):
                X_batch = X.next_batch(batch_size)
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.X: X_batch[0]})
                if i % 250 == 0:
                    print("Train [%d/%d] loss = %9.4f" % (i+1, rounds, loss))

                    summary = self.sess.run(self.merged, feed_dict={self.X: X_batch[0]})
                    self.writer.add_summary(summary=summary, global_step=(epoch+1)*rounds+i+1)

            # validation
            [loss] = self.sess.run([self.loss], feed_dict={self.X: X_valid.images})
            print("Valid loss = %9.4f" % loss)

            # testing
            if X_test:
                [loss] = self.sess.run([self.loss], feed_dict={self.X: X_test.images})
                print("Test loss = %9.4f" % loss)

        self.saver.save(self.sess, "./Models/MNIST_"+str(self.code_nodes)+".ckpt")  # keep the model


    def predict(self, X):
        """
        Test for img to encode and decode
        :param X: the img matrix to go through encode and decode process
        :return: encoded code and decode image matrix
        """
        code, y_pred = self.sess.run([self.code, self.y_], feed_dict={self.X: X})
        return code, y_pred

    def predict_from_code(self, codes):

        return self.sess.run(self.y_, feed_dict={self.code: codes})



if __name__ == '__main__':


    layer_nodes = [1024, 512, 256, 64]
    encode_nodes = 2

    # load = False
    load = "./Models/MNIST_2.ckpt.meta"

    if not load: # train from scratch
        train_data, valid_data, test_data, input_shape, label_shape = load_mnist()

        auto_encoder = AutoEncoder(input_shape, layer_nodes, encode_nodes, learning_rate=0.00017, alpha=5, scale=0.053)

        auto_encoder.fit(train_data, valid_data, test_data, total_sample=train_data.images.shape[0], epochs= 20, batch_size=8)

        figures = 15

        # set figures
        fig, axes = plt.subplots(2, figures, figsize=(figures*1.1, 3), # set size and nums of figures
                                 subplot_kw={'xticks': [], 'yticks': []}) # remove the ticks
        fig.subplots_adjust(hspace=0.3, wspace=0.05)

        fig.suptitle("MNIST Auto Encoder (Code Size="+str(encode_nodes)+")", size=15) # title for whole figure

        for i in range(figures):
            img = np.reshape(test_data.images[i, :], (28, 28)) # re-shape to image like

            c, y_pred = auto_encoder.predict(np.reshape(test_data.images[i, :], (1, 28*28)))
            img_pred = np.reshape(y_pred, (28, 28)) # re-shape to image like

            print(c)

            axes[0][i].imshow(img, cmap=plt.get_cmap('gray'))

            axes[1][i].imshow(img_pred, cmap=plt.get_cmap('gray'))

        plt.savefig('res_img/mnist_'+str(encode_nodes)+'.png')
        plt.show()

    else: # load pre train
        auto_encoder = AutoEncoder(load=load)

        codes = [[0.1, 0.1], [0.1, 100], [0.1, 200],
                 [100, 0.1], [100, 100], [100, 200],
                 [200, 0.1], [200, 100], [200, 200]
                ]

        # set figures
        fig, axes = plt.subplots(3, 3, figsize=(3 * 1.5, 3*1.5),  # set size and nums of figures
                                 subplot_kw={'xticks': [], 'yticks': []})  # remove the ticks

        fig.subplots_adjust(hspace=0.5, wspace=0.05)

        fig.suptitle("MNIST Decoder(Size=" + str(encode_nodes) + ")", size=15)  # title for whole figure

        decode_res = auto_encoder.predict_from_code(codes)
        for i, res in enumerate(decode_res):
            img_pred = np.reshape(res, (28, 28))
            axes[int(i / 3)][i % 3].set_title(codes[i])
            axes[int(i/3)][i%3].imshow(img_pred, cmap=plt.get_cmap('gray'))
        plt.savefig('res_img/mnist_decode_' + str(encode_nodes) + '.png')
        plt.show()