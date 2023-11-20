import pickle
import tensorflow as tf
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

img = image.load_img('images/9992.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors([normalized_result])

print(indices)

# Display 5 images
for file in indices[0][1:6]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output', cv2.resize(temp_img, (512, 512)))
    cv2.waitKey(0)

    import matplotlib.pyplot as plt
import cv2

# Display the query image
query_img = cv2.imread('images/9338.jpg')
plt.subplot(1, 6, 1)
plt.imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
plt.title('Query Image')
plt.axis('off')

# Display the top 5 recommended images
for i, file in enumerate(indices[0][1:6], 2):
    temp_img = cv2.imread(filenames[file])
    plt.subplot(1, 6, i)
    plt.imshow(cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Rec. Image {i-1}')
    plt.axis('off')

plt.show()

