from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import keras
import gensim
from keras.models import load_model

app = Flask(__name__)
dibaca_h5_model = load_model("bisaDibaca.h5")
dibaca_bin_model = gensim.models.Word2Vec.load("bisaDibaca.bin")
w2v_h5_model = load_model("w2v.h5")
w2v_bin_model = gensim.models.Word2Vec.load("w2v.bin")

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
        if bisa :
            mitos_fakta_vectors = get_avg_word_vectors(cleanKata, w2v_bin_model)
            predictions = w2v_h5_model.predict(mitos_fakta_vectors)
            hasil = "Fakta" if predictions[0]>=0.5 else "Mitos"
        else :
            hasil = "Tidak Bisa Dibaca Bang"
        return render_template('index.html', userInput = hasil)
    return render_template('index.html', userInput = None)

if __name__ == '__main__':
    app.run(debug=True)