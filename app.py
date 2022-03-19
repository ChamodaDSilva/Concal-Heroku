from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import joblib
#import pickle

app = Flask(__name__)
model = joblib.load('concrete-predicter.joblib')

# model = pickle.load(open('concrete-predicter.pkl','rb'))

# @app.route('/')
# def home():
#     return "hello World"
#
# @app.route('/predict',methods=['POST'])
# def predict():
#     inputs=request.form.get('inputs')
#     newDataArr = inputs.split()
#     data = [[float(newDataArr[0]), float(newDataArr[1]), float(newDataArr[2]), float(newDataArr[3]), float(newDataArr[4]),float(newDataArr[5]), float(newDataArr[6]), float(newDataArr[7])]]
#     prediction = np.array2string(model.predict(data)[0])
#     print(prediction)
#
#     return jsonify({'result':str(prediction)})
@app.route('/', methods=['POST'])
def feedModel():

    print("backend started")

    newdata=request.form.get('inputs')
    print(newdata)
    newDataArr = newdata.split()
    data = [
        [float(newDataArr[0]), float(newDataArr[1]), float(newDataArr[2]), float(newDataArr[3]), float(newDataArr[4]),
         float(newDataArr[5]), float(newDataArr[6]), float(newDataArr[7])]]
    prediction = np.array2string(model.predict(data)[0])
    print(prediction)
    return jsonify({"output":str(prediction)})
if __name__=='__main__':
    app.run(debug=True)


#
#
# if __name__ == '__main__':
#     app.run(port=5000, debug=True)  # change to '0.0.0.0' your IPv4 address
