from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Загрузка модели и списка признаков
model, all_features = joblib.load('ensemble_model.pkl')
features = all_features

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    proba = None
    if request.method == 'POST':
        try:
            values = [float(request.form[feature]) for feature in features]
            df = pd.DataFrame([values], columns=features)
            pred = model.predict(df)[0]
            prob = model.predict_proba(df)[0][1]
            prediction = 'Активная sgRNA' if pred == 1 else 'Неактивная sgRNA'
            proba = f'{prob:.2f}'
        except Exception as e:
            prediction = f'Ошибка: {e}'
    return render_template('index.html', features=features, prediction=prediction, proba=proba)

if __name__ == '__main__':
    app.run(port=8508)

from pyngrok import ngrok
public_url = ngrok.connect(8508)
print("Ссылка на приложение:", public_url)
