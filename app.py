from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os

# Cargar el modelo .h5
model = tf.keras.models.load_model("modelo_lstm.h5")

# Inicializar la aplicación Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos de la solicitud
        data = request.get_json(force=True)
        
        # Convertir los datos en un formato adecuado para el modelo
        # Añade un valor más si el modelo lo requiere (por ejemplo, un cero)
        input_data = np.array([[
            data['edad'],
            data['estatura'],
            data['peso'],
            data['dosis_quimioterapia'],
            0  # Añade este valor si el modelo necesita cinco características
        ]]).reshape(1, 1, 5)  # Cambia la forma aquí según lo que el modelo espera
        
        # Realizar la predicción
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        # Devolver la clase predicha
        return jsonify({'predicted_class': int(predicted_class)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Utilizar el puerto proporcionado por Railway o el puerto 5000 por defecto
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
