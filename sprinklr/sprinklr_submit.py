import os

os.environ['KERAS_BACKEND'] = "theano"
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu1"
from keras.callbacks import EarlyStopping, ModelCheckpoint
from deepmoji.model_def import deepmoji_architecture
from sprinklr_global import path, max_nb_words, max_sequence_length
from sprinklr_generator import generate_data,generate_validation_data

model = deepmoji_architecture(nb_classes=64, nb_tokens=max_nb_words, maxlen=max_sequence_length)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
early_stopping = EarlyStopping(patience=2, verbose=1)
check_pointer = ModelCheckpoint(
    filepath=path + "model-{epoch:02d}", save_best_only=True)

model.fit_generator(generate_data(),steps_per_epoch=2500,epochs=10,verbose=1,
                    callbacks=[early_stopping,check_pointer],
                    validation_data=generate_validation_data(),validation_steps=98,max_queue_size=3)

