import numpy as np
import data_helpers
import tensorflow as tf
from tensorflow.contrib import learn
# Data Preparation
# ==================================================



tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "../cnn-text-classification-tf/data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "../cnn-text-classification-tf/data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

# Build vocabulary
max_document_length = max([len(sentence.split(" ")) for sentence in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))
vocab_size = len(vocab_processor.vocabulary_)

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

# Load pre-train file
# filename = '../cnn-text-classification-tf/data/GoogleNews-vectors-negative300.bin'
filename = '../cnn-text-classification-tf/data/glove.6B.200d.txt'
print("Load word2vec file {}\n".format(filename))
initW = np.random.uniform(-0.25,0.25,(vocab_size, FLAGS.embedding_dim))
with open(filename, "r") as f:
    line = f.readline()
    while line:
        word = line.split()[0]
        vec = line.split()[1:]
        # CHeck if the word is in the dictionary
        idx = vocab_processor.vocabulary_.get(word)
        # If exist assign vector, else check next word
        if idx != 0:
            initW[idx] = np.array(vec)

print("Build ")
W = tf.Variable(tf.constant(0.0,
                shape=[vocab_size, embedding_dim]), # or tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                trainable=False, name="W")

sess = tf.Session()

# Way 1
sess.run(cnn.W.assign(initW))

# Way2
# embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
# embedding_init = W.assign(embedding_placeholder)
# sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})
