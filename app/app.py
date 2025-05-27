from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model, features = joblib.load("model.pkl")

def extract_features(seq):
    return pd.DataFrame([{  # пример
        "GC_content": (seq.upper().count("G") + seq.upper().count("C")) / len(seq),
        "has_TTTT": int("TTTT" in seq),
    }])

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        seq = request.form["sequence"]
        features_df = extract_features(seq)
        prediction = model.predict(features_df)[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
