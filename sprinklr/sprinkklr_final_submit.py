import os

os.environ['KERAS_BACKEND'] = "theano"
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda1"

from deepmoji.model_def import deepmoji_architecture
from sprinklr_global import max_nb_words
from sprinklr_global import path, max_sequence_length, train_files
from sprinklr_data import prepare_data_from_json

model = deepmoji_architecture(nb_classes=64, nb_tokens=max_nb_words, maxlen=max_sequence_length)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

for epoch in range(10):
    for file_number in range(len(train_files)):
        X_train, Y_train = prepare_data_from_json(train_files[file_number])
        print("Loaded X_train: " + str(X_train.shape) + " Y_train: " + str(Y_train.shape))
        model.fit(X_train, Y_train, batch_size=256, epochs=1, verbose=1)
        model.save_weights(path + "-" + str(epoch) + "-" + str(file_number))
