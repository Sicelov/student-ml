from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and features list
with open('student_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array([features])
    prediction = model.predict(final_features)

    output = "PASS" if prediction[0] == 1 else "FAIL"
    return render_template('index.html', prediction_text=f'Student will: {output}')


if __name__ == "__main__":
    app.run(debug=True)
