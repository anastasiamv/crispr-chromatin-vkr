from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
import threading

app = Flask(__name__)
model, used_features, scaler = joblib.load("ensemble_model.pkl")

def extract_features_from_sequence(seq):
    seq = seq.upper()
    features = {
        "GC_content": (seq.count("G") + seq.count("C")) / len(seq),
        "has_TTTT": int("TTTT" in seq),
        "has_GGGG": int("GGGG" in seq),
        "has_TTT[ACG]": int(any(seq[i:i+4].startswith("TTT") and seq[i+3] in "ACG" for i in range(len(seq)-3))),
        "Tm_20mer": 64.9 + 41 * (seq.count("G") + seq.count("C") - 16.4) / len(seq),
        "entropy_20mer": -sum(seq.count(nuc)/len(seq) * np.log2(seq.count(nuc)/len(seq)) for nuc in "ATGC" if seq.count(nuc) > 0),
        "hairpin_score": 3,
        "has_hairpin": 0,
        "exact_hairpin": 0,
        "GC_clamp": int(seq[-1] in "GC"),
        "melting_seed": 68.4 + 41 * (seq[10:17].count("G") + seq[10:17].count("C") - 16.4) / 7,
        "seed_gc": (seq[10:17].count("G") + seq[10:17].count("C")) / 7,
        "seed_entropy": -sum(seq[10:17].count(n)/7 * np.log2(seq[10:17].count(n)/7) for n in "ATGC" if seq[10:17].count(n) > 0),
        "GC_chromatin_interaction": 0.5,
        "Normalized TSS of primary TSS, 5'": 0.2,
        "Normalized TSS of primary TSS, 3'": 0.2,
        "Normalized TSS of secondary TSS, 5'": 0.1,
        "Normalized TSS of secondary TSS, 3'": 0.1,
        "Normalized ATAC values": 0.6,
        "Normalized RNA values": 0.4,
        "Normalized methylation values": 0.4,
        "chromatin_signal": 0.0,
        "cCRE_present": 1,
        "cCRE_distance": 80,
        "cCRE_distance_log": np.log(80),
        "bed_overlap": 1,
    }
    return pd.DataFrame([{k: features[k] for k in used_features}])

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    proba = None
    guide = ""
    if request.method == "POST":
        try:
            guide = request.form["sequence"]
            df = extract_features_from_sequence(guide)
            pred = model.predict(df)[0]
            prob = model.predict_proba(df)[0][1]
            prediction = "Активная sgRNA" if pred == 1 else "Неактивная sgRNA"
            proba = f"{prob:.2f}"
        except Exception as e:
            prediction = f"Ошибка: {e}"
    return render_template("seq_input.html", prediction=prediction, proba=proba, sequence=guide)

if __name__ == "__main__":
    threading.Thread(target=app.run, kwargs={"port": 8513}).start()
