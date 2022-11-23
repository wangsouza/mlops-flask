import os
import pickle
from textblob import TextBlob
from flask_basicauth import BasicAuth
from flask import Flask, request, jsonify

columns = ['tamanho', 'ano', 'garagem']
model = pickle.load(open('../../models/model.sav', 'rb'))

# Dar nomes
app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app)


# Definir rotas (endpoints)
@app.route('/')
def home():
    return 'Minha primeira API'


@app.route('/sentiment/<phrase>')
@basic_auth.required
def sentiment(phrase):
    tb = TextBlob(phrase)
    tb_en = tb.translate(from_lang='pt-br', to='en')
    polarity = tb_en.sentiment.polarity
    return 'polarity: {}'.format(polarity)

@app.route('/quotation/', methods=['POST'])
@basic_auth.required
def quotation():
    data = request.get_json()   # traz o json que o usuário envia
    data_input = [data[col] for col in columns] # Varre a lista de colunas
    price = model.predict([data_input]) # Prediz uma lista de colunas
    return jsonify(price=price[0])  # Facilita a entrega no formato json

# O Debug True faz com que quando o script for alterado e for salvo o flask 
# identifica automaticamente fez uma alteração e faz o restart.
if __name__ == '__main__':
    # Usa-se host=0.0.0.0 quando for fazer o deploy da aplicação em vários
    # ambientes diferentes. A aplicação vai escutar chamadas dentro do docker,
    # local ou append
    app.run(debug=True, host='0.0.0.0')


