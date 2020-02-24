from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pathlib

model = keras.models.load_model('lib/fruits_model.h5')

print('Loading dataset...')
data_dir = pathlib.Path('./fruits_dataset_test')
class_names = np.array([ item.name for item in data_dir.glob('*') if item.name != '.DS_Store' ])
image_count = len(list(data_dir.glob('*/*.jpg')))

image_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     shuffle=True,
                                                     target_size=(224, 224),
                                                     batch_size=25,
                                                     classes = list(class_names))

test_images, test_labels = next(test_data_gen)

predictions = model.predict(test_images)

plt.figure(figsize=(10, 10))
for i in range(len(predictions)):
  item_title = class_names[np.argmax(predictions[i])]
  plt.subplot(5, 5, i+1)
  plt_title = plt.title(item_title)
  plt.imshow(test_images[i])
  plt.axis('off')
  if (np.argmax(predictions[i]) != np.argmax(test_labels[i])):
    plt_title.set_color('red')

plt.show()