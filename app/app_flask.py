from flask import Flask, request, render_template
import joblib
import pandas as pd
import threading

app = Flask(__name__)
model, features, scaler = joblib.load("ensemble_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    proba = None
    if request.method == "POST":
        try:
            values = [float(request.form[feature]) for feature in features]
            df = pd.DataFrame([values], columns=features)
            df_scaled = scaler.transform(df)
            pred = model.predict(df_scaled)[0]
            prob = model.predict_proba(df_scaled)[0][1]
            prediction = "Активная sgRNA" if pred == 1 else "Неактивная sgRNA"
            proba = f"{prob:.2f}"
        except Exception as e:
            prediction = f"Ошибка: {e}"
    return render_template("index.html", features=features, prediction=prediction, proba=proba)

if __name__ == "__main__":
    threading.Thread(target=app.run, kwargs={"port": 8524}).start()
