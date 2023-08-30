from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import keras
import gensim
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
from keras.models import load_model
import requests
from sklearn.model_selection import train_test_split

app = Flask(__name__)
dibaca_h5_model = load_model("bisaDibaca.h5")
dibaca_bin_model = gensim.models.Word2Vec.load("bisaDibaca.bin")
mata_h5_model = load_model("temaMata.h5")
mata_bin_model = gensim.models.Word2Vec.load("temaMata.bin")
w2v_h5_model = load_model("w2v.h5")
w2v_bin_model = gensim.models.Word2Vec.load("w2v.bin")


def train_model():
    url = "http://127.0.0.1:5000/sentences"
    summary = []
    response = requests.get(url)
    data = response.json()
    for i in range(0,3):
        if i == 0:
            tidak = data['reads'][0]['0']
            iya = data['reads'][0]['1']
        if i == 1:
            tidak = data['contents'][0]['0']
            iya = data['contents'][0]['1']
        if i == 2:
            tidak = data['categories'][0]['0']
            iya = data['categories'][0]['1']
        labelledTidak = [(kalimat, 0) for kalimat in tidak]
        labelledIya = [(kalimat, 1) for kalimat in iya]
        trainData = labelledTidak + labelledIya
        cleanData = [(clean_and_lower(kalimat), label) for kalimat, label in trainData]
        sentences = [data for data, label in cleanData]
        labels = [label for data, label in cleanData]
        X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.3, random_state=42)
        word2vec = word2vec_model(X_train) # ajarin model sentence word2vec
        X_train_avg = get_avg_word_vectors(X_train, word2vec)
        X_test_avg = get_avg_word_vectors(X_test,word2vec)
        if i == 0:
            model = build_model_read(input_dim=X_train_avg.shape[1])
            result = model.fit(X_train_avg, np.array(y_train), batch_size=5, epochs=25, verbose=1, validation_data=(X_test_avg, np.array(y_test)))
            summary.append({'read' : [result.history['loss'][-1],
                                      result.history['accuracy'][-1],
                                      result.history['val_loss'][-1],
                                      result.history['val_accuracy'][-1]]})
            model.save("bisaDibaca.h5")
            word2vec.save("bisaDibaca.bin")
        if i == 1:
            model = build_model_content(input_dim=X_train_avg.shape[1])
            result = model.fit(X_train_avg, np.array(y_train), batch_size=5, epochs=20, verbose=1, validation_data=(X_test_avg, np.array(y_test)))
            summary.append({'content' : [result.history['loss'][-1],
                                         result.history['accuracy'][-1],
                                         result.history['val_loss'][-1],
                                         result.history['val_accuracy'][-1]]})
            model.save("temaMata.h5")
            word2vec.save("temaMata.bin")
        if i == 2:
            model = build_model_category(input_dim=X_train_avg.shape[1])
            result = model.fit(X_train_avg, np.array(y_train), batch_size=3, epochs=20, verbose=1, validation_data=(X_test_avg, np.array(y_test)))
            summary.append({'category' : [result.history['loss'][-1],
                                          result.history['accuracy'][-1],
                                          result.history['val_loss'][-1],
                                          result.history['val_accuracy'][-1]]})
            model.save("w2v.h5")
            word2vec.save("w2v.bin")
    return summary

def build_model_read(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_model_content(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_model_category(input_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def clean_and_lower(text):
    cleaned_text = ''.join(char.lower() for char in text if char.isalnum() or char.isspace())
    return cleaned_text

def word2vec_model(sentences):
    words = [sentence.split() for sentence in sentences]
    model = gensim.models.Word2Vec(words, vector_size=100, window=5, min_count=1, workers=4)
    return model

def get_avg_word_vectors(sentences, model):
    avg_vectors = []
    for sentence in sentences:
        vectors = []
        for word in sentence.split():
            if word in model.wv:
                vectors.append(model.wv[word])
        if vectors:
            avg_vectors.append(np.mean(vectors, axis=0))
        else:
            avg_vectors.append(np.zeros(model.vector_size))
    return np.array(avg_vectors)

@app.route('/train', methods=['POST'])
def train():
    summary = train_model()
    return render_template('addTrain.html', hasil = summary)


# Route for the root page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST' :
        user_input = list(request.form['userInput'])
        cleanKata = [''.join(clean_and_lower(kalimat) for kalimat in user_input)]
        bisa_dibaca_vector = get_avg_word_vectors(cleanKata, dibaca_bin_model)
        bisa_dibaca_prediksi = dibaca_h5_model.predict(bisa_dibaca_vector)
        bisa = True if bisa_dibaca_prediksi[0]>=0.5 else False
        print("Bisa Dibaca :", bisa_dibaca_prediksi)
        if bisa :
            tema_mata_vector = get_avg_word_vectors(cleanKata, mata_bin_model)
            tema_mata_prediksi = mata_h5_model.predict(tema_mata_vector)
            mata = True if tema_mata_prediksi[0]>=0.5 else False
            print("Tema Mata :", tema_mata_prediksi)
            if mata :
                mitos_fakta_vectors = get_avg_word_vectors(cleanKata, w2v_bin_model)
                predictions = w2v_h5_model.predict(mitos_fakta_vectors)
                hasil = "Fakta" if predictions[0]>=0.5 else "Mitos"
                print("Fakta Mitos :", predictions)
            else :
                hasil = "Bukan tentang mata"
        else :
            hasil = "Inputan Sementara Tidak Dikenal"
        return render_template('index.html', userInput = hasil)
    return render_template('index.html', userInput = None)

if __name__ == '__main__':
    app.run(debug=True)