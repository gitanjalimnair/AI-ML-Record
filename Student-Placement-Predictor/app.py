from flask import Flask, render_template, request
import pickle
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load trained model
with open("placement_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    iq = float(request.form["iq"])
    prev_sem = float(request.form["prev_sem"])
    cgpa = float(request.form["cgpa"])
    academic = float(request.form["academic"])
    internship = int(request.form["internship"])
    extracurricular = float(request.form["extracurricular"])
    communication = float(request.form["communication"])
    projects = int(request.form["projects"])

    features = np.array([[
        iq,
        prev_sem,
        cgpa,
        academic,
        internship,
        extracurricular,
        communication,
        projects
    ]])

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)

    confidence = round(max(probability[0]) * 100, 2)

    if prediction == 1:
        status = "LIKELY TO BE PLACED"
        status_class = "success"
    else:
        status = "NOT LIKELY TO BE PLACED"
        status_class = "failure"

    return render_template(
    "result.html",
    prediction=status,
    confidence=confidence,
    status_class=status_class
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)