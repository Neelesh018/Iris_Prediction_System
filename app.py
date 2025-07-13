from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

clf = None
label_encoder = None

def load_model():
    global clf, label_encoder

    data = {
        'sepal_length': [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9, 7.0, 6.4, 6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2, 6.3, 3.3, 5.8, 7.1, 6.3, 5.8, 7.6, 4.9, 7.3, 6.7],
        'sepal_width': [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1, 3.2, 3.2, 3.1, 2.3, 2.8, 2.8, 3.3, 2.4, 2.9, 2.7, 3.3, 3.0, 2.7, 3.0, 2.9, 2.7, 3.0, 2.5, 2.9, 2.5],
        'petal_length': [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5, 4.7, 4.5, 4.9, 4.0, 4.6, 4.5, 4.7, 3.3, 4.6, 3.9, 6.0, 5.0, 5.1, 5.9, 5.6, 5.1, 6.6, 4.5, 6.3, 5.8],
        'petal_width': [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 1.4, 1.5, 1.5, 1.3, 1.5, 1.3, 1.6, 1.0, 1.3, 1.4, 2.5, 1.9, 1.9, 2.1, 1.8, 1.9, 2.1, 1.7, 1.8, 1.8],
        'species': ['setosa']*10 + ['versicolor']*10 + ['virginica']*10
    }

    df = pd.DataFrame(data)
    X = df.iloc[:, 0:4]
    y = df['species']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y_encoded)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        global clf, label_encoder
        if clf is None or label_encoder is None:
            load_model()

        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        if not (0 < sepal_length < 10 and 0 < sepal_width < 10 and
                0 < petal_length < 10 and 0 < petal_width < 10):
            return jsonify({'success': False, 'error': 'Values must be between 0 and 10 cm'})

        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = clf.predict(features)[0]
        confidence = max(clf.predict_proba(features)[0]) * 100
        species = label_encoder.inverse_transform([prediction])[0]

        return jsonify({
            'success': True,
            'species': f"Iris {species.title()}",
            'confidence': round(confidence, 1)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': f'Prediction error: {str(e)}'})

if __name__ == '__main__':
    load_model()
    app.run(debug=True)

