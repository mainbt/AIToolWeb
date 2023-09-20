import cv2
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

app = Flask(__name__)

model = tf.keras.models.load_model("saved_model/MobileNetV2_retrain")

@app.route('/', methods=['POST'])
def home():
    data = request.files['file']
    result = process_data(data)
    return jsonify(result)
    

def process_data(data):

    map_dict = {0: 'cardboard',
                1: 'compost',
                2: 'glass',
                3: 'metal',
                4: 'paper',
                5: 'plastic',
                6: 'trash',}

    if data is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(data.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(opencv_image,(256,256))

        resized = mobilenet_v2_preprocess_input(resized)
        img_reshape = resized[np.newaxis,...]  
        prediction = model.predict(img_reshape).argmax()

        return { "result": map_dict[prediction] }

if __name__ == '__main__':
    app.run()