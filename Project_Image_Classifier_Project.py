

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from PIL import Image

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

import tensorflow_hub as hub    
from workspace_utils import active_session

# ### Label Mapping


import json
with open('label_map.json', 'r') as f:
    class_names = json.load(f)


# ## Load the Dataset

Load the dataset with TensorFlow Datasets.
splits = ['train[:80%]', 'train[80%:]', 'test']
dataset, dataset_info = tfds.load('oxford_flowers102', as_supervised=True, with_info=True, split=splits)

# Create a training set, a validation set and a test set.
training_set, validation_set, testing_set = dataset


# ## Explore the Dataset

# Get the number of examples in each set from the dataset info.
print(f"Number of examples in a training set: {len(training_set)}")
print(f"Number of examples in a validation set: {len(validation_set)}")
print(f"Number of examples in a testing set: {len(testing_set)}")
# Get the number of classes in the dataset from the dataset info.
print(f"Number of classes: {dataset_info.features['label'].num_classes}")


# Print the shape and corresponding label of 3 images in the training set.
for image, label in training_set.take(3):
    print(f"The image shape: {image.shape}, Label: {label}")

# Plot 1 image from the training set. Set the title 
# of the plot to the corresponding image label. 
for image, label in training_set.take(1):
    image = image.numpy().squeeze()
    label = label.numpy()

plt.title(label)
plt.imshow(image)
plt.colorbar()
plt.show()




# Plot 1 image from the training set. Set the title 
# of the plot to the corresponding class name. 
for image, label in training_set.take(1):
    image = image.numpy().squeeze()
    label = label.numpy()

label_name = class_names.get(str(label), f"Label {label}")
plt.title(label_name)
plt.imshow(image)
plt.colorbar()
plt.show()


# ## Create Pipeline

# Create a pipeline for each set.
def normalize(image, label):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

batch_size = 64

training_batches = training_set.cache().shuffle(len(training_set) // 4).map(normalize).batch(batch_size).prefetch(1)

validation_batches = validation_set.cache().map(normalize).batch(batch_size).prefetch(1)

testing_batches = testing_set.cache().map(normalize).batch(batch_size).prefetch(1)



# Build and train your network.


with active_session():
    url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
    model = tf.keras.Sequential([
         hub.KerasLayer(url, input_shape=(224, 224, 3)),
        tf.keras.layers.Dense(512, activation = 'relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(dataset_info.features['label'].num_classes, activation = 'softmax'),
    ])
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    history = model.fit(training_batches, epochs=3, validation_data=validation_batches)


# Plot the loss and accuracy values achieved during training for the training and validation set.
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()



# Print the loss and accuracy values achieved on the entire test set.
with active_session():    
    loss, accuracy = model.evaluate(testing_batches)
    print(f"Accuracy: {accuracy}")
    print(f"Loss: {loss}")


# ## Save the Model
# 

#  Save trained model as a Keras model.
model.save('./classifier.keras')


# ## Load the Keras Model

import tensorflow_hub as hub
model = tf.keras.models.load_model('./classifier.keras', custom_objects={'KerasLayer': hub.KerasLayer})




# Create the process_image function
def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255.0
    return image.numpy()



image_path = './test_images/hard-leaved_pocket_orchid.jpg'
im = Image.open(image_path)
test_image = np.asarray(im)

processed_test_image = process_image(test_image)

fig, (ax1, ax2) = plt.subplots(figsize=(10,10), ncols=2)
ax1.imshow(test_image)
ax1.set_title('Original Image')
ax2.imshow(processed_test_image)
ax2.set_title('Processed Image')
plt.tight_layout()
plt.show()


# Create the predict function
def predict(image_path, model, top_k):
    image = Image.open(image_path)
    image_np = np.asarray(image)
    
    processed_image = process_image(image_np)
    
    processed_image = np.expand_dims(processed_image, axis=0)
    
    predictions = model.predict(processed_image)
    probabilities = predictions[0]
    
    top_k_indices = np.argsort(probabilities)[-top_k:][::-1]
    top_k_probs = probabilities[top_k_indices]
    
    top_k_classes = [str(idx) for idx in top_k_indices]
    
    return top_k_probs, top_k_classes


image_path = './test_images/cautleya_spicata.jpg'
top_k = 5

probs, classes = predict(image_path, model, top_k)
mapped_names = [class_names.get(str(cls), str(cls)) for cls in classes]

img = Image.open(image_path)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.imshow(img)
ax1.axis('off')
ax1.set_title('Input Image')

y_pos = np.arange(len(classes))
ax2.barh(y_pos, probs, align='center')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(mapped_names)
ax2.invert_yaxis()  
ax2.set_xlabel('Probability')
ax2.set_title('Top 5 Predictions')

plt.tight_layout()
plt.show()





