from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import keras
import gensim
from keras.models import load_model

app = Flask(__name__)
dibaca_h5_model = load_model("bisaDibaca.h5")
dibaca_bin_model = gensim.models.Word2Vec.load("bisaDibaca.bin")
mata_h5_model = load_model("temaMata.h5")
mata_bin_model = gensim.models.Word2Vec.load("temaMata.bin")
w2v_h5_model = load_model("w2v.h5")
w2v_bin_model = gensim.models.Word2Vec.load("w2v.bin")

def train_model():
    pass

def clean_and_lower(text):
    cleaned_text = ''.join(char.lower() for char in text if char.isalnum() or char.isspace())
    return cleaned_text

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