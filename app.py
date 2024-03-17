import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## load the model
reg_model=pickle.load(open('reg_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    data_array=np.array(list(data.values())).reshape(1,-1)
    output=reg_model.predict(data_array)
    print(output[0])
    return jsonify(output[0])

if __name__=="__main__":
    app.run(debug=True)

