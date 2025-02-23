from get_input_args import get_input_args
import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255.0
    return image.numpy()

def predict(image_path, model, top_k = 5):
    image = Image.open(image_path)
    image = np.asarray(image)
    
    processed_image = process_image(image)
    
    processed_image = np.expand_dims(processed_image, axis=0)
    predictions = model.predict(processed_image)
    probabilities = predictions[0]

    top_k_indices = np.argsort(probabilities)[-top_k:][::-1]
    top_k_probs = probabilities[top_k_indices]
    
    return top_k_probs, top_k_indices


def main():
    in_arg = get_input_args()

    saved_model = in_arg.saved_model
    image_path = in_arg.image_path
    
    model = tf.keras.models.load_model(saved_model, custom_objects={'KerasLayer': hub.KerasLayer})
    top_k_prob, top_k_index = predict(image_path, model, in_arg.top_k)
    if in_arg.category_names:
        with open(in_arg.category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_k_classes = [cat_to_name.get(str(index), str(index)) for index in top_k_index]
    else:
        top_k_classes = [str(index) for index in top_k_index]
    
        print("Top {} Predictions:".format(in_arg.top_k))
    for prob, classes in zip(top_k_prob, top_k_classes):
        print("Class: {:>15}   Probability: {:.4f}".format(classes, prob))

if __name__ == "__main__":
    main()
