import os
import pickle
# from textblob import TextBlob
# from flask_basicauth import BasicAuth
import keras
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask import render_template
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


max_words = 5000
max_len = 200

columns = ["Neutral", "Negative", "Positive"]
model = keras.models.load_model("models/best_model_test1.hdf5")
tokenizer = pickle.load(open('models/tokenizer.sav', 'rb'))


# Dar nomes
app = Flask(__name__)
# app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
# app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

# basic_auth = BasicAuth(app)


# Definir rotas (endpoints)
@app.route('/')
def index():
    return render_template('form.html')


@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment():

    text_area = request.form['name']

    if text_area and request.method == "POST":

        sequence = tokenizer.texts_to_sequences([text_area])
        test = pad_sequences(sequence, maxlen=max_len)
        result = columns[np.around(model.predict(test), decimals=0).argmax(axis=1)[0]]
        label = 'Sentiment: {}'.format(result)
        return jsonify({'name': label})

    return jsonify({'error': 'Campo vazio! Insira uma nova entrada.'})


if __name__ == '__main__':
    # Usa-se host=0.0.0.0 quando for fazer o deploy da aplicação em vários
    # ambientes diferentes. A aplicação vai escutar chamadas dentro do docker,
    # local ou append
    app.run(debug=True, host='0.0.0.0')
