from flask import Flask, render_template, request
import joblib

model = joblib.load('model.pkl')  
vectorizer = joblib.load('vectorizer.pkl') 

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sms = request.form['sms']
        sms_vectorized = vectorizer.transform([sms])  
        prediction = model.predict(sms_vectorized)[0]
        result = 'Spam' if prediction == 1 else 'Not Spam'
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
