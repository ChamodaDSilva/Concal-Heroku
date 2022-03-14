
from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('concrete-predicter.joblib')


@app.route('/', methods=['POST', 'GET'])
def feedModel():
    print("backend started")
    newdata = request.get_json('input')
    print(newdata)
    newDataArr = newdata['input'].split()
    data = [
        [float(newDataArr[0]), float(newDataArr[1]), float(newDataArr[2]), float(newDataArr[3]), float(newDataArr[4]),
         float(newDataArr[5]), float(newDataArr[6]), float(newDataArr[7])]]
    prediction = np.array2string(model.predict(data)[0])
    print(prediction)

    return jsonify(prediction)


if __name__ == '__main__':
    app.run(port=5000, debug=True)  # change to '0.0.0.0' your IPv4 address
