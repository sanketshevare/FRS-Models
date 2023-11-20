from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import pickle
import base64
import io
from PIL import Image


app = Flask(__name__)

# Load image embeddings and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

# Nearest Neighbors model
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)

def get_recommendations(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    print("Result")
    print(result)
    normalized_result = result / norm(result)
    print(normalized_result)
    distances, indices = neighbors.kneighbors([normalized_result])
    return indices[0][1:6]

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_image

def save_base64_image(base64_string, output_path):
    image_data = base64.b64decode(base64_string)
    with open(output_path, "wb") as image_file:
        image_file.write(image_data)


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        img = request.files['image']
        img.save('uploaded_image.jpg')
        recommendations = get_recommendations('uploaded_image.jpg')
        print("--------")
        print(recommendations)
        print(type(recommendations))
        recommended_images = [filenames[idx] for idx in recommendations]
        base64_images = []
        i=0
        for r in recommended_images:
            print(".........")
            print(r)
            base64_image = image_to_base64(r)
            base64_images.append(base64_image)
            output_path = str(i)+".jpg"
            save_base64_image(base64_image, output_path)
            i+=1


       
        print(base64_images)
        return {'recommendations': base64_images}
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') 