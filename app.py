from flask import Flask, render_template, request
import pickle

# Load the ML model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

# Detection function
def detect(input_text):
    vectorized_text = tfidf.transform([input_text])
    result = model.predict(vectorized_text)
    return "Plagiarism Detected" if result[0] == 1 else "No Plagiarism"

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['text']
        prediction = detect(input_text)
        return render_template('index.html', result=prediction, input_text=input_text)

# Run the app

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
