import os
import pickle
import random

import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

from deepmoji.filter_input import read_wanted_emojis
from deepmoji.filter_utils import extract_emojis
from sprinklr_global import path, path_to_emoji, path_to_train, max_sequence_length, train_files, validation_files
from sprinklr_preprocess import pre_process

wanted_emojis = read_wanted_emojis(path_to_emoji)
wanted_emojis_index = pd.Index(wanted_emojis)
tokenizer = pickle.load(open(path + "tokenizer", "rb"))


def generate_data():
    file_number = 0
    data_train = []
    label_train = []
    while 1:
        print("Loading file: " + train_files[file_number])
        with open(os.path.join(path_to_train, train_files[file_number])) as json_file:
            data_pre_process = json_file.readlines()
            data_pre_process = [pre_process(x) for x in data_pre_process]
            data_pre_process = filter(lambda y: y != '', data_pre_process)
            random.shuffle(data_pre_process)
            for i in range(len(data_pre_process)):
                text = data_pre_process[i]
                emojis = extract_emojis(text, wanted_emojis)
                emojis = np.unique(emojis)
                for l in range(len(emojis)):
                    text = text.replace(emojis[l], '')
                for j in range(len(emojis)):
                    data_train.append(text)
                    label_train.append(wanted_emojis_index.get_loc(emojis[j]))
                if i % 1024 == 1023 or i == len(data_pre_process) - 1:
                    sequences = tokenizer.texts_to_sequences(data_train)
                    X_train = pad_sequences(sequences, maxlen=max_sequence_length)
                    Y_train = np.eye(64)[label_train]
                    print("Loaded " + str(len(X_train)) + " records from" + train_files[file_number])
                    yield (X_train, Y_train)
                    data_train = []
                    label_train = []
        file_number = (file_number + 1) % len(train_files)


def generate_validation_data():
    file_number = 0
    data_val = []
    label_val = []
    while 1:
        print("Loading Validation file: " + validation_files[file_number])
        with open(os.path.join(path_to_train, validation_files[file_number])) as json_file:
            data_preprocess = json_file.readlines()
            data_preprocess = [pre_process(x) for x in data_preprocess]
            data_preprocess = filter(lambda y: y != '', data_preprocess)
            random.shuffle(data_preprocess)
            for i in range(len(data_preprocess)):
                text = data_preprocess[i]
                emojis = extract_emojis(text, wanted_emojis)
                emojis = np.unique(emojis)
                for l in range(len(emojis)):
                    text = text.replace(emojis[l], '')
                for j in range(len(emojis)):
                    data_val.append(text)
                    label_val.append(wanted_emojis_index.get_loc(emojis[j]))
                if (i % 1024 == 1023 or i == len(data_preprocess) - 1):
                    sequences = tokenizer.texts_to_sequences(data_val)
                    X_val = pad_sequences(sequences, maxlen=max_sequence_length)
                    Y_val = np.eye(64)[label_val]
                    print("Loaded Validation " + str(len(X_val)) + " records from" + validation_files[file_number])
                    yield (X_val, Y_val)
                    data_val = []
                    label_val = []
        file_number = (file_number + 1) % len(validation_files)