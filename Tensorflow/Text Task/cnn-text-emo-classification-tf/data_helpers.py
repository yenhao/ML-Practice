import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels(train_data_file, test_data_file, label_hot_dict = {'joy':         [1,0,0,0,0,0,0,0],
                                                                            'surprise':    [0,1,0,0,0,0,0,0],
                                                                            'trust':       [0,0,1,0,0,0,0,0],
                                                                            'anticipation':[0,0,0,1,0,0,0,0],
                                                                            'fear':        [0,0,0,0,1,0,0,0],
                                                                            'sadness':     [0,0,0,0,0,1,0,0],
                                                                            'disgust':     [0,0,0,0,0,0,1,0],
                                                                            'anger':       [0,0,0,0,0,0,0,1]
                                                                            }):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    train_examples = list(open(train_data_file, "r").readlines())
    train_examples = [sentence.strip() for sentence in train_examples]
    test_examples  = list(open(test_data_file, "r").readlines())
    test_examples  = [sentence.strip() for sentence in test_examples]
    print('Train sentences & labels preparing...')
    train_text = []
    train_labels = []
    test_text = []
    test_labels = []
    for sent in train_examples:
        if len(sent.split('\t')) !=2:
            continue
        train_text.append(clean_str(sent.split('\t')[1]))
        train_labels.append(label_hot_dict[sent.split('\t')[0]])

    print('Testing sentences & labels preparing...')
    for sent in test_examples:
        if len(sent.split('\t')) == 1:
            continue
        test_text.append(clean_str(sent.split('\t')[0]))
        test_labels.append(label_hot_dict[sent.split('\t')[1]])
    print(train_text[0])
    print(train_labels[0])
    return [train_text, np.array(train_labels), test_text, np.array(test_labels)]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
