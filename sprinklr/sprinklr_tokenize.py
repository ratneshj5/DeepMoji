import os
import pickle

import keras
from keras.preprocessing.text import Tokenizer, maketrans

from sprinklr_global import train_files, tokenization_file_number
from sprinklr_global import path_to_train, max_nb_words, path
from sprinklr_preprocess import pre_process

final_texts = []
with open(os.path.join(path_to_train, train_files[tokenization_file_number])) as json_file:
    content = json_file.readlines()
    content = [pre_process(x) for x in content]
    content = filter(lambda x: x != '', content)
    for i in range(len(content)):
        text = content[i]
        final_texts.append(text)
    print(len(final_texts))


def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    if lower: text = text.lower()
    if type(text) == unicode:
        translate_table = {ord(c): ord(t) for c, t in zip(filters, split * len(filters))}
    else:
        translate_table = maketrans(filters, split * len(filters))
    text = text.translate(translate_table)
    seq = text.split(split)
    return [i for i in seq if i]


keras.preprocessing.text.text_to_word_sequence = text_to_word_sequence

tokenizer = Tokenizer(num_words=max_nb_words)
tokenizer.fit_on_texts(final_texts)
print("Dumping tokenizer at " + path + "tokenizer")
pickle.dump(tokenizer, open(path + "tokenizer", "wb"))
