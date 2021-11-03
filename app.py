from flask import Flask, request, url_for, redirect, render_template

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import decmodel
from decmodel import results
from decmodel import cls_report
from decmodel import acc
from decmodel import df
from decmodel import tr
from decmodel import dtc

from linearmodel import b0,b1
from linearmodel import rmse
from linearmodel import r2
 


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/decision')
def decision():
    return render_template('decisionmain.html')

@app.route('/prediction', methods = ["post"])
def prediction():
    return render_template('decisionpredictionpage.html')

@app.route("/predict", methods = ["post"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = decmodel.predict(features)
    return render_template("decisionpredictionpage.html", prediction_text = "The flower species is {}".format(prediction))

@app.route("/result", methods = ["POST"])
def result():
    cr_array= list(cls_report.split(" "))
    return render_template("result.html",
    result=results,
    cr=cr_array,
    accuracy=" Accuracy is : {}".format(acc))

@app.route('/decparameters', methods = ["post"])
def decparameters():
    return render_template('decpara.html')


@app.route("/train", methods = ["POST"])
def train():
    pickle.dump(dtc, open("model.pkl", "wb"))



    #print("Hello Karthik")

    return render_template("train.html")



# ----------------------------------------LINEAR MODELS--------------------------------------------------------

@app.route('/linear')
def linear():
    return render_template('linearmain.html')

@app.route('/linear_prediction', methods = ["post"])
def linear_prediction():
    return render_template('linearpredictionpage.html')

@app.route("/linear_result", methods = ["POST"])
def linear_result():
    return render_template("linearresult.html", 
    cr="coefficient of regression is: {}".format(b0),
    root="Root mean square error is:{}".format(rmse),
    r2="R2 score is : {}".format(r2))
    



# @app.route('/cool_form', methods=['GET', 'POST'])
# def cool_form():
#     if request.method == 'POST':
#         # do stuff when the form is submitted

#         # redirect to end the POST handling
#         # the redirect can be to the same route or somewhere else
#         return redirect(url_for('index'))

#     # show the form, it wasn't submitted
#     return render_template('cool_form.html')

if __name__ == "__main__":
    app.run(debug=True)