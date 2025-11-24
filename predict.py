import tensorflow as tf
import numpy as np
import cv2

BATCH_SIZE = 1
DATASET_PATH = "dataset"
MODEL_PATH = "model/image_model.keras"
IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)
print(model.summary())
print("Model loaded from", MODEL_PATH)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    shuffle=True,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
print("Class names:", class_names)

def predict_image(image):
    img = cv2.imread(image)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    class_index = np.argmax(predictions[0])
    return class_names[class_index]