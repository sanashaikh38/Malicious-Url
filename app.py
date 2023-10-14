from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load(r"E:\phishing project\PHISHINNG\model\your_trained_model.pkl")

# Assuming you have a tfidf_vectorizer defined somewhere
tfidf_vectorizer = TfidfVectorizer()

# Load your dataset into a DataFrame (assuming your data is in a CSV file)
df = pd.read_csv('E:\phishing project\PHISHINNG\datasets\malicious_phish.csv')

# Split the data
X = df['url']
y = df['type']

X_tfidf = tfidf_vectorizer.fit_transform(X)

# Train a simple classifier (e.g., Multinomial Naive Bayes)
classifier = MultinomialNB()
classifier.fit(X_tfidf, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        new_url = request.form['url']
        new_url_tfidf = tfidf_vectorizer.transform([new_url])
        predicted_label = classifier.predict(new_url_tfidf)

        return render_template('index.html', result=f" {predicted_label[0]}")

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
