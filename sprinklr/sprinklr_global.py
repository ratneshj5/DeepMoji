import os
from random import randint

base_path = '/Users/ratnesh/Documents/Sprinklr/dataset/'

# TODO fix path
path_to_train = base_path + 'arabic_korean/'
path_to_validation = base_path + 'arabic_korean/'
path_to_emoji = '/Users/ratnesh/Documents/Sprinklr/DeepMoji/data/filtering/wanted_emojis.csv'

path = base_path + 'arabic/'

path_to_model = path + "model/model-0-30"
path_to_sentiment_model = path + "sentiment/model/emoji_to_sentiment_model_dec15.pkl"
path_to_sentiment_validation = path + "sentiment/validation/deepmoji.xlsx"

language = 'ar'

max_nb_words = 50000
max_sequence_length = 100

train_files = [pos_json for pos_json in os.listdir(path_to_train) if pos_json.endswith('.json')]
validation_files = [pos_json for pos_json in os.listdir(path_to_validation) if pos_json.endswith('.json')]

tokenization_file_number = randint(0, len(train_files) - 1)
