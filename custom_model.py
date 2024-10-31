import pickle
import pandas as pd
import numpy as np
import re
import os
import json
import requests
import http.client
from sklearn import preprocessing
# Import pad_sequences from keras.preprocessing.sequence for TensorFlow 2.7.0
from keras.preprocessing.sequence import pad_sequences


API_TOKEN = os.environ.get('API_TOKEN')
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit_transform(pd.DataFrame(['ORGANIZATION', 'PERSON']))
maxlen = 10

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def preprocess_text(sentence):
    sentence = re.sub('[^a-zA-Z]', '', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", '', sentence)
    sentence = re.sub(r'\s+', '', sentence)
    return sentence

def get_padded_sequence(text):
    text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([text])[0]
    padded_sequence = pad_sequences([sequence], padding='post', maxlen=maxlen)
    return padded_sequence.tolist()

def get_mlflow_response(padded_sequence):
    conn = http.client.HTTPSConnection(os.environ.get('DATABRICKS_BASELINK'))
    payload = json.dumps({"instances": padded_sequence})
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_TOKEN}'
    }
    conn.request("POST", os.environ.get('CUSTOM_ENDPOINT'), payload, headers)
    res = conn.getresponse()
    data = res.read().decode("utf-8")
    data_dict = json.loads(data)
    return data_dict["predictions"]

def get_class_and_confidence(prediction):
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    confidence_ratio = np.max(prediction, axis=1)[0]
    predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
    return predicted_class, confidence_ratio * 100

def get_custom_model_result(text):
    padded_sequence = get_padded_sequence(text)
    print(padded_sequence)
    prediction = get_mlflow_response(padded_sequence)
    predicted_class, confidence_ratio = get_class_and_confidence(prediction)
    return predicted_class, confidence_ratio
