import os

import pickle

import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
os.environ['KERAS_BACKEND'] = "theano"
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda1"

from deepmoji.model_def import deepmoji_architecture
from sprinklr.sprinklr_data import prepare_predict_data_from_excel,prepare_predict_data_from_sample
from sprinklr.sprinklr_global import *
os.system("pip install xlrd")
os.system("pip install xgboost==0.6a1")
X_val, Y_val = prepare_predict_data_from_excel(path_to_sentiment_validation)
# X_val, Y_val = prepare_predict_data_from_sample()

model = deepmoji_architecture(nb_classes=64, nb_tokens=max_nb_words, maxlen=max_sequence_length)
model.load_weights(path_to_model)
emojis = model.predict(X_val)

features = []

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]


for row_features in emojis:
    ind_top = top_elements(row_features, 5)
    sum_top_5 = sum(row_features[ind_top])
    row_features = list(row_features)
    row_features.append(sum_top_5)
    features.append(row_features)

model = pickle.load(open(path_to_sentiment_model, "rb"))
predictions = model.predict(features)
mapping = {'neutral':0,'positive':1,'negative':2}
y = []
pred = []
for i in range(len(Y_val)):
    if str(Y_val[i]) != 'nan':
        y.append(mapping[Y_val[i].lower()])
        pred.append(mapping[predictions[i].lower()])
print classification_report(y, pred,target_names=['neutral','positive','negative'])
print(confusion_matrix(y, pred))