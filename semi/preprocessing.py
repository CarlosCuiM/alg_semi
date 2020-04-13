# -*- coding: UTF-8 -*-


import numpy as np


def shuffle(x, labels):

    randomize = np.arange(len(labels))
    np.random.shuffle(randomize)
    x_shuffle = x[randomize]
    labels_shuffle = labels[randomize]
    return x_shuffle, labels_shuffle


def encoder(seq, unlabelled_mark=" "):

    valid_labels = set(seq).difference(unlabelled_mark)
    encoder_dict = dict(zip(valid_labels, range(len(valid_labels))))
    encoded_seq = np.zeros(len(seq), dtype='int32')
    for i in range(len(seq)):
        if seq[i] == unlabelled_mark:
            encoded_seq[i] = 0
        else:
            encoded_seq[i] = encoder_dict[seq[i]]
    return encoded_seq, encoder_dict


def decoder(encoded_seq, encoder_dict):

    decoder_dict = dict(zip(encoder_dict.values(), encoder_dict.keys()))
    decode_seq = []
    for i in encoded_seq:
        decode_seq.append(decoder_dict[i])
    return decode_seq

