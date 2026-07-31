from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("car_price_model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        present_price = float(request.form["present_price"])
        kms_driven = float(request.form["kms_driven"])
        fuel_type = int(request.form["fuel_type"])
        seller_type = int(request.form["seller_type"])
        transmission = int(request.form["transmission"])
        owner = int(request.form["owner"])
        car_age = int(request.form["car_age"])

        features = np.array([[present_price,
                              kms_driven,
                              fuel_type,
                              seller_type,
                              transmission,
                              owner,
                              car_age]])

        prediction = model.predict(features)[0]

        return render_template(
            "index.html",
            prediction_text=f"Estimated Selling Price: ₹ {prediction:.2f} Lakhs"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {e}"
        )


if __name__ == "__main__":
    app.run(debug=True)