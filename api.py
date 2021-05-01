# We now need the json library so we can load and export json data
import json
import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import pandas as pd
from joblib import load
from sklearn import preprocessing

from flask import Flask

# Set environnment variables
MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE_LDA = os.environ["MODEL_FILE_LDA"]
MODEL_FILE_NN = os.environ["MODEL_FILE_NN"]
MODEL_PATH_LDA = os.path.join(MODEL_DIR, MODEL_FILE_LDA)
MODEL_PATH_NN = os.path.join(MODEL_DIR, MODEL_FILE_NN)

# Loading LDA model
print("Loading model from: {}".format(MODEL_PATH_LDA))
inference_lda = load(MODEL_PATH_LDA)

# loading Neural Network model
print("Loading model from: {}".format(MODEL_PATH_NN))
inference_NN = load(MODEL_PATH_NN)

# Creation of the Flask app
app = Flask(__name__)

# API 1
# Flask route so that we can serve HTTP traffic on that route
@app.route('/line/<Line>')
# Get data from json and return the requested row defined by the variable Line
def line(Line):
    with open('./test.json', 'r') as jsonfile:
       file_data = json.loads(jsonfile.read())
    # We can then find the data for the requested row and send it back as json
    return json.dumps(file_data[Line])
    

# API 2
# Flask route so that we can serve HTTP traffic on that route
@app.route('/prediction/<int:Line>',methods=['POST', 'GET'])
# Return prediction for both Neural Network and LDA inference model with the requested row as input
def prediction(Line):
    data = pd.read_json('./test.json')
    data_test = data.transpose()
    X = data_test.drop(data_test.loc[:, 'Line':'# Letter'].columns, axis = 1)
    X_test = X.iloc[Line,:].values.reshape(1, -1)
    
    clf_lda = load(MODEL_PATH_LDA)
    prediction_lda = clf_lda.predict(X_test)
    
    clf_nn = load(MODEL_PATH_NN)
    prediction_nn = clf_nn.predict(X_test)
    
    return {'prediction LDA': int(prediction_lda), 'prediction Neural Network': int(prediction_nn)}

# API 3
# Flask route so that we can serve HTTP traffic on that route
@app.route('/score',methods=['POST', 'GET'])
# Return classification score for both Neural Network and LDA inference model from the all dataset provided
def score():

    data = pd.read_json('./test.json')
    data_test = data.transpose()
    y_test = data_test['# Letter'].values
    X_test = data_test.drop(data_test.loc[:, 'Line':'# Letter'].columns, axis = 1)
    
    clf_lda = load(MODEL_PATH_LDA)
    score_lda = clf_lda.score(X_test, y_test)
    
    clf_nn = load(MODEL_PATH_NN)
    score_nn = clf_nn.score(X_test, y_test)
    
    return {'Score LDA': score_lda, 'Score Neural Network': score_nn}

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
    
