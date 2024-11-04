from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os

# Cargar el modelo .h5
model = tf.keras.models.load_model("modelo_lstm.h5")

# Definir el mapeo de índice a clase sin el prefijo "Clase"
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

# Inicializar la aplicación Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos de la solicitud
        data = request.get_json(force=True)
        
        # Convertir los datos en un formato adecuado para el modelo
        # Añadimos un valor ficticio para completar las 5 características
        input_data = np.array([
            data['edad'],
            data['estatura'],
            data['peso'],
            data['dosis_quimioterapia'],
            0  # Valor adicional para cumplir con el requisito de 5 características
        ]).reshape(1, 1, 5)  # Cambiamos el reshape a (1, 1, 5) para cumplir con el input esperado
        
        # Realizar la predicción
        prediction = model.predict(input_data)
        predicted_index = np.argmax(prediction, axis=1)[0]
        
        # Convertir el índice a la clase original
        predicted_class = class_mapping.get(predicted_index, "Clase desconocida")
        
        # Devolver la clase predicha
        return jsonify({'predicted_class': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Utilizar el puerto proporcionado por Railway o el puerto 5000 por defecto
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
