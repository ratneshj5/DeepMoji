# This Python file uses the following encoding: utf-8

import os
import pickle

import numpy as np
import pandas as pd

os.environ['KERAS_BACKEND'] = "theano"
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu1"
from deepmoji.filter_utils import extract_emojis
from deepmoji.filter_input import read_wanted_emojis
from keras.preprocessing.sequence import pad_sequences
from sprinklr_global import path, max_sequence_length, path_to_train, path_to_emoji
from sprinklr_preprocess import pre_process, convert_to_unicode

wanted_emojis = read_wanted_emojis(path_to_emoji)
wanted_emojis_index = pd.Index(wanted_emojis)
tokenizer = pickle.load(open(path + "tokenizer", "rb"))

def prepare_data_from_json(file_to_load):
    final_texts = []
    final_labels = []
    print("Loading file: " + file_to_load)
    with open(os.path.join(path_to_train, file_to_load)) as json_file:
        content = json_file.readlines()
        content = [pre_process(x) for x in content]
        content = filter(lambda y: y != '', content)
        for i in range((len(content))):
            text = content[i]
            emojis = extract_emojis(text, wanted_emojis)
            emojis = np.unique(emojis)
            for l in range(len(emojis)):
                text = text.replace(emojis[l], '')
            for j in range(len(emojis)):
                final_texts.append(text)
                final_labels.append(wanted_emojis_index.get_loc(emojis[j]))
    data = prepare_texts(final_texts)
    final_labels = np.eye(64)[final_labels]
    return data,final_labels

def prepare_data_from_excel(path_to_load):
    final_texts = []
    final_labels = []
    data = pd.read_excel(path_to_load)
    content = data["Message"].tolist()
    content = [convert_to_unicode(x) for x in content]
    content = filter(lambda y: y != '', content)
    for i in range((len(content))):
        text = content[i]
        emojis = extract_emojis(text, wanted_emojis)
        emojis = np.unique(emojis)
        for l in range(len(emojis)):
            text = text.replace(emojis[l], '')
        for j in range(len(emojis)):
            final_texts.append(text)
            final_labels.append(wanted_emojis_index.get_loc(emojis[j]))
    data = prepare_texts(final_texts)
    final_labels = np.eye(64)[final_labels]
    return data, final_labels

def prepare_predict_data_from_excel(path_to_load):
    data = pd.read_excel(path_to_load)
    labels = data["QA Team"].tolist()
    texts = data["Message"].tolist()

    texts = [convert_to_unicode(x) for x in texts]
    data = prepare_texts(texts)
    return data,labels

def prepare_predict_data_from_sample():
    # labels = data["QA Team"].tolist()
    # texts = data["Message"].tolist()
    texts = ['احبك','اللعنة عليك']
    labels = ['positive','negative']
    texts = [convert_to_unicode(x) for x in texts]
    data = prepare_texts(texts)
    return data,labels

def prepare_texts(final_texts):
    sequences = tokenizer.texts_to_sequences(final_texts)
    data = pad_sequences(sequences, maxlen=max_sequence_length)
    return data