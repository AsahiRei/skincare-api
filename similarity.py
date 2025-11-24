import tensorflow as tf
import numpy as np
import cv2
import os
import requests
from io import BytesIO

DATASET_PATH = "dataset"
MODEL_PATH = "model/image_model.keras"
IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)

feature_model = tf.keras.Sequential(model.layers[:-1])

def load_image(image_path_or_url):
    if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
        response = requests.get(image_path_or_url)
        if response.status_code != 200:
            raise ValueError(f"Could not fetch image: {image_path_or_url}")
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(image_path_or_url)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path_or_url}")
    
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_embedding(image_path_or_url):
    img = load_image(image_path_or_url)
    embedding = feature_model.predict(img)
    return embedding[0]

def cosine_similarity_np(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_most_similar(query_image):
    query_emb = get_embedding(query_image)
    most_similar = None
    highest_similarity = -1
    for root, _, files in os.walk(DATASET_PATH):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                emb = get_embedding(img_path)
                sim = cosine_similarity_np(query_emb, emb)
                if sim > highest_similarity:
                    highest_similarity = sim
                    most_similar = img_path
                    
    relative_path = os.path.relpath(most_similar, DATASET_PATH)
    display_name = os.path.basename(os.path.dirname(relative_path))
    return display_name, highest_similarity

def get_similarity(query_image):
    most_similar_image, score = find_most_similar(query_image)
    return most_similar_image, score