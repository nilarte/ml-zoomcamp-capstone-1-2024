import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'rf.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('leaveornot')

@app.route('/predict', methods=['POST'])
def predict():
    employee = request.get_json()

    X = dv.transform([employee])
    y_pred = model.predict_proba(X)[0, 1]
    leaveornot = y_pred >= 0.5

    result = {
        'leaveornot_probability': float(y_pred),
        'leaveornot': bool(leaveornot)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)