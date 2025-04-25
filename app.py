import os
import time
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

# ─── Inicialización de la app ────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # habilita CORS para todas las rutas

# ─── Cargar el modelo .h5 y warm-up ───────────────────────────────────────────
model = tf.keras.models.load_model("modelo_lstm.h5")
print("⚡ Realizando warm-up del modelo…")
try:
    # un batch de ceros con shape (batch=1, timesteps=1, features=5)
    _ = model.predict(np.zeros((1, 1, 5)))
    print("✅ Warm-up completado")
except Exception as e:
    print("❌ Error en warm-up:", e)

# ─── Mapeo de índices a clases ────────────────────────────────────────────────
class_mapping = {
    0: "IA1", 1: "IA2", 2: "IB",  3: "IIA", 4: "IIIA",
    5: "IIIB",6: "IIIC",7: "IVA", 8: "IVB"
}

# ─── Endpoint de salud ───────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

# ─── Endpoint de predicción ─────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        print("Datos recibidos:", data)

        # Construir array de entrada con 5 características
        input_data = np.array([
            data['edad'],
            data['estatura'],
            data['peso'],
            data['dosis_quimioterapia'],
            0  # valor ficticio
        ]).reshape(1, 1, 5)
        print("Shape de entrada:", input_data.shape)

        # Medir tiempo de inferencia
        t0 = time.time()
        prediction = model.predict(input_data)
        dt = time.time() - t0
        print(f"🔍 Inferencia en {dt:.3f}s")

        print("➤ Salida cruda:", prediction)
        predicted_index = int(np.argmax(prediction, axis=1)[0])
        predicted_class = class_mapping.get(predicted_index, "Clase desconocida")
        print("↪️ Clase final:", predicted_class)

        return jsonify({'predicted_class': predicted_class})

    except Exception as e:
        print("❌ Error en predict():", e)
        return jsonify({'error': str(e)}), 400

# ─── Arranque ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
