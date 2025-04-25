import os
import time
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

# ─── Inicialización de la app ────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ─── Carga del modelo y warm-up ───────────────────────────────────────────────
model = tf.keras.models.load_model("modelo_lstm.h5")
print("⚡ Realizando warm-up del modelo…")
try:
    _ = model.predict(np.zeros((1, 1, 5)))
    print("✅ Warm-up completado")
except Exception as e:
    print("❌ Error en warm-up:", e)

# ─── Mapeo de índices a clases ────────────────────────────────────────────────
class_mapping = {
    0: "IA1",
    1: "IA2",
    2: "IB",
    3: "IIA",
    4: "IIIA",
    5: "IIIB",
    6: "IIIC",
    7: "IVA",
    8: "IVB"
}

# ─── Endpoint de salud ───────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

# ─── Stub GET para discovery en APEX ─────────────────────────────────────────
@app.get("/predict")
def predict_get():
    return jsonify({"predicted_class": "IA1"})
# ──────────────────────────────────────────────────────────────────────────────

# ─── Endpoint de predicción ─────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        print("Datos recibidos:", data)

        # Construir array de entrada
        input_data = np.array([
            data['edad'],
            data['estatura'],
            data['peso'],
            data['dosis_quimioterapia'],
            0
        ]).reshape(1, 1, 5)
        print("Shape de entrada:", input_data.shape)

        # Tiempo de inferencia
        t0 = time.time()
        prediction = model.predict(input_data)
        dt = time.time() - t0
        print(f"🔍 Inferencia terminada en {dt:.3f}s")

        print("➤ Salida cruda:", prediction)
        idx = int(np.argmax(prediction, axis=1)[0])
        cls = class_mapping.get(idx, "Clase desconocida")
        print("↪️ Clase final:", cls)

        return jsonify({'predicted_class': cls})

    except Exception as e:
        print("❌ Error en predict():", e)
        return jsonify({'error': str(e)}), 400

# ─── Arranque ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
