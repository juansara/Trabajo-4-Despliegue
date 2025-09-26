from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

modelo = joblib.load("modelo_spotify_pipeline_xgb.pkl")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "API is running"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        df = pd.DataFrame([data])

        pred = modelo.predict(df)[0]

        return jsonify({
            "prediction": int(pred),
            "meaning": "Popular" if pred == 1 else "Not popular"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)