from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib

# Prepare the dataset

print('Loading dataset...')
data_dir = pathlib.Path('./fruits_dataset_train')
class_names = np.array([ item.name for item in data_dir.glob('*') if item.name != '.DS_Store' ])
print(class_names)
image_count = len(list(data_dir.glob('*/*.jpg')))

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     shuffle=True,
                                                     batch_size=512,
                                                     target_size=(224, 224),
                                                     classes = list(class_names))

train_images, train_labels = next(train_data_gen)

# Create the model

model = keras.Sequential([
  keras.layers.Conv2D(64, 3, activation="relu", input_shape=(224, 224, 3)),
  keras.layers.MaxPool2D(2),
  keras.layers.Conv2D(32, 3, activation="relu"),
  keras.layers.MaxPool2D(2),
  keras.layers.Conv2D(32, 3, activation="relu"),
  keras.layers.MaxPool2D(2),
  keras.layers.Flatten(),
  keras.layers.Dense(512, activation="relu"),
  keras.layers.Dropout(0.5),
  keras.layers.Dense(64, activation="relu"),
  keras.layers.Dropout(0.5),
  keras.layers.Dense(2, activation="softmax")
])

model.compile(optimizer='adam',
              loss="categorical_crossentropy",
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

model.save('lib/fruits_model.h5')
