# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS           # opcional: elimina si no lo necesitas
import tensorflow as tf
import numpy as np
import os
import logging

# ── Configuración mínima de logging ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Carga del modelo (.h5) ──────────────────────────────────────────────────────
MODEL_PATH = "modelo_lstm.h5"
logger.info(f"Cargando modelo desde {MODEL_PATH} …")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)  # compile=False evita warnings
logger.info("Modelo cargado correctamente")

# ── Mapeo índice → clase ────────────────────────────────────────────────────────
class_mapping = {
    0: "IA1", 1: "IA2", 2: "IB", 3: "IIA",
    4: "IIIA", 5: "IIIB", 6: "IIIC",
    7: "IVA", 8: "IVB"
}

# ── Inicializar Flask ──────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)                               # opcional: habilita CORS

# ── Endpoints ───────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """Endpoint de prueba para verificar que el contenedor está vivo."""
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)  # fuerza carga de JSON
        logger.info(f"Datos recibidos: {data}")

        # Validar claves requeridas
        required = {"edad", "estatura", "peso", "dosis_quimioterapia"}
        if not required.issubset(data):
            faltantes = required.difference(data)
            return jsonify({"error": f"Faltan campos: {', '.join(faltantes)}"}), 400

        # Construir ndarray (1, 1, 5)
        input_data = np.array([
            data["edad"],
            data["estatura"],
            data["peso"],
            data["dosis_quimioterapia"],
            0  # valor de relleno
        ], dtype=np.float32).reshape(1, 1, 5)

        # Predicción
        preds = model.predict(input_data)
        idx = int(np.argmax(preds, axis=1)[0])
        predicted_class = class_mapping.get(idx, "Clase desconocida")

        return jsonify({"predicted_class": predicted_class})

    except Exception as exc:
        logger.exception("Error en /predict")
        return jsonify({"error": str(exc)}), 500


# ── Main local (Railway usará Gunicorn) ─────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
