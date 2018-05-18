from deepmoji.model_def import deepmoji_architecture
from sprinklr_global import path, max_nb_words, max_sequence_length

model = deepmoji_architecture(nb_classes=64, nb_tokens=max_nb_words, maxlen=max_sequence_length)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.load_weights(path + "model-best")
model.predict()
