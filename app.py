from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import keras
from transformers import BertTokenizer, TFBertModel

app = Flask(__name__)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = TFBertModel.from_pretrained('bert-base-uncased')
# model = keras.models.load_model('nlpModel.keras')

def clean_and_lower(text):
    cleaned_text = ''.join(char.lower() for char in text if char.isalnum() or char.isspace())
    return cleaned_text

# Route for the root page
@app.route('/', methods=['GET', 'POST'])
def home():
    # if request.method == 'POST' :
    #     user_input = request.form['userInput']
    #     cleanKata = clean_and_lower(user_input)
        
    #     inputsCleanKata = tokenizer(cleanKata, padding=True, truncation=True, return_tensors="tf")
    #     input_idsCleanKata = inputsCleanKata["input_ids"].numpy()
    #     butuh = np.zeros(panjangPred-panjangTeks, dtype=int)

    #     manipulasi = np.append(input_idsCleanKata[0],butuh)
    #     manipulasi = np.array([manipulasi])

    #     panjangTeks = input_idsCleanKata.shape[1]
    #     panjangPred = 85 #iki kudu dinamis ngko
    #     hasil = model.predict(manipulasi)

    #     return render_template('index.html', userInput = hasil)
    return render_template('index.html', userInput = None)

if __name__ == '__main__':
    app.run(debug=True)