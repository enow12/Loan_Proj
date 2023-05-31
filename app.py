from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
app = Flask(__name__)
filename = 'file_loan.pkl'
#model = pickle.load(open(filename, 'rb'))
model = joblib.load(filename)
#model = joblib.load(filename)
@app.route('/')
def index(): 
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    Credit_History = request.form['credit_history']
    Applicant_Income = request.form['applicant_income']
    Loan_Amount = request.form['loan_amount']
    Coapplicant_Income = request.form['coapplicant_income']

    
      
    pred = model.predict(np.array([[Credit_History, Applicant_Income, Loan_Amount, Coapplicant_Income ]]))
    print(pred)
    return render_template('index.html', predict=str(pred))


if __name__ == '__main__':
    app.run
