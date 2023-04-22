from flask import Flask,render_template,redirect,flash,request
from flask_cors import CORS, cross_origin
import pickle
import numpy as np
app=Flask(__name__)

CORS(app)
@app.route("/")
def home():
    return render_template('index.html')
@app.route("/predict",methods=['POST','GET'])
def predict():
    try:
        if request.method=='POST':
            age=request.form.get('age')
            gender=request.form.get("gender")
            family_history=request.form.get('family_history')
            benifits=request.form.get('benefits')
            care_options=request.form.get('care_options')
            anonymity=request.form.get('anonymity')
            leave=request.form.get('leave')
            work_interfere=request.form.get('work_interfere')
            feature_list=np.expand_dims(np.array([age,gender,family_history,benifits,care_options,anonymity,leave,work_interfere]),axis=0)
            feature_list=feature_list.astype(int)
            with open('boostmodel.pkl','rb') as f:
                model=pickle.load(f)
            prediction=model.predict(feature_list)
            print(prediction)
            return "{}".format(prediction)
        else:
            return render_template('index.html')
    except Exception as e:
        print(e)

if __name__=='__main__':
    app.run(host='0.0.0.0', port=1234, debug=True)
