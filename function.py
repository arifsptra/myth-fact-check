from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

class Function:
    def preprocess_text(self, text):
        cleaned_text = ''.join(char.lower() for char in text if char.isalnum() or char.isspace())
        return cleaned_text

    def findResult(self, dataAll, text):
        vectorizer = TfidfVectorizer()
        rfc = RandomForestClassifier()

        processed_review = [(self.preprocess_text(words), category) for words, category in dataAll]
        attribut, label = zip(*processed_review)
        attributVec = vectorizer.fit_transform(attribut)
        rfc.fit(attributVec, label)

        textVec = vectorizer.transform([text])
        prediksi = rfc.predict(textVec)[0]
        return 'Fakta' if prediksi=='fakta' else 'Mitos'