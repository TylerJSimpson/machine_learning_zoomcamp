import pickle
from flask import Flask, request, jsonify

model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('credit-risk')

@app.route('/predict', methods=['POST'])
def predict():
    applicant = request.get_json()

    X_applicant = dv.transform([applicant])
    y_pred_applicant = model.predict_proba(X_applicant)[:, 1]
    approval_probability = float(y_pred_applicant)
    approval = approval_probability >= 0.5

    result = {
        'approval_probability': approval_probability,
        'approval': approval
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
