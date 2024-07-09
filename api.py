from flask import Flask, request, jsonify
import numpy as np
import onnxruntime as ort
from PIL import Image
import io

app = Flask(__name__)

# Cargar el modelo ONNX
model_path = "modeloOnnx.onnx"
ort_session = ort.InferenceSession(model_path)

# Lista de clases
class_names = ['abeto', 'bamboo', 'palmera', 'almendro', 'redmaple']

def preprocess_image(image):
    # Redimensionar la imagen a 224x224 y convertirla a un array de NumPy
    image = image.resize((224, 224))
    image = np.array(image).astype('float32')
    image = np.expand_dims(image, axis=0)  # Añadir una dimensión para el batch
    image = image / 255.0  # Normalizar la imagen
    return image

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    image = preprocess_image(image)

    # Realizar la inferencia
    inputs = {ort_session.get_inputs()[0].name: image}
    outputs = ort_session.run(None, inputs)
    predictions = outputs[0]
    
    # Obtener la clase predicha y su probabilidad
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]
    predicted_class_probability = predictions[0][predicted_class_index]
    
    return jsonify({
        'predicted_class': predicted_class_name,
        'probability': float(predicted_class_probability)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
