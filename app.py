from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
from numpy.core.arrayprint import array_str

app = Flask(__name__,static_url_path='/static')


filename = 'model'
model = joblib.load(filename)


@app.route("/")
@app.route("/home")
def home():

    return render_template("index.html") 

@app.route("/predict")
def predict():

    return render_template("predict.html")


@app.route("/result", methods=['POST'])
def result():
    age = request.form['age']
    bmi = request.form['bmi']
    married = request.form['ever_married']
    gender = request.form['gender']
    glucose = request.form['glucose']
    residence = request.form['residence_type']
    smoking = request.form['smoking_status']
    work = request.form['work_status']
    hypertension = request.form['hypertension']
    heart = request.form['heart']

    input_data = np.zeros(21)

    gender_switch = {
        "0": 0,
        "1": 1
    }

    age_switch = {
        "0": 7,
        "1": 8,
        "2": 9,
        "3": 10,
        "4": 11
    }

    married_switch = {
        "0": 0,
        "1": 1
    }

    hypertension_switch = {
        "0": 0,
        "1": 1
    }

    heart_switch = {
        "0": 0,
        "1": 1
    }

    residence_switch = {
        "0": 0,
        "1": 1
    }

    smoking_switch = {
        "0": 17,
        "1": 18,
        "2": 19,
        "3": 20
    }

    work_switch = {
        "0": 12,
        "1": 13,
        "2": 14,
        "3": 15,
        "4": 16
    }

    input_data[0] = gender_switch[gender]
    input_data[age_switch[age]] = 1
    input_data[6] = bmi
    input_data[5] = glucose
    input_data[3] = married_switch[married]
    input_data[1] = hypertension_switch[hypertension]
    input_data[2] = heart_switch[heart]
    input_data[4] = residence_switch[residence]
    input_data[ smoking_switch[smoking] ] = 1
    input_data[ work_switch[work] ] = 1

    result = model.predict_proba([input_data])
    chanceToNotHappen = round(result[0][0]*100,2)
    chanceToHappen = round(result[0][1]*100,2)

    
    return render_template('result.html',chanceToNotHappen = chanceToNotHappen,chanceToHappen = chanceToHappen )


@app.route("/helpline")
def helpline():
    return render_template('helpline.html')

@app.route("/about")
def about():
    return render_template('about.html')




if __name__ == "__main__":
    app.run(debug=True,port=8000)
