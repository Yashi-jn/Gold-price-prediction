from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("gold_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        spx = float(request.form["spx"])
        uso = float(request.form["uso"])
        slv = float(request.form["slv"])
        eur_usd = float(request.form["eur_usd"])

        data = np.array([[spx, uso, slv, eur_usd]])
        pred = model.predict(data)
        prediction = round(pred[0], 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
