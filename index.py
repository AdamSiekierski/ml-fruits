from tensorflow import keras
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import pathlib

app = Flask(__name__)
model = None
class_names = None

def prepare_image(image, size):
  if image.mode != "RGB":
    image = image.convert("RGB")

  image = image.resize(size)
  image = keras.preprocessing.image.img_to_array(image)
  image = np.expand_dims(image, axis=0)

  return image

def load_model():
  global model
  model = keras.models.load_model('lib/fruits_model.h5')

  global class_names
  data_dir = pathlib.Path('./fruits_dataset_train')
  class_names = np.array([ item.name for item in data_dir.glob('*') if item.name != '.DS_Store' ])

@app.route('/', methods=["GET"])
def root():
  return jsonify({ "message": 'hello' })

@app.route('/predict', methods=["POST"])
def predict():
  if request.method == "POST":
    if request.files.get("image"):
      image = request.files["image"].read()
      image = Image.open(io.BytesIO(image))
      image = prepare_image(image, size=(224, 224))

      prediction = model.predict(image)[0]
      result = class_names[np.argmax(prediction)]
      return jsonify({ "message": f"Predicred, that given image is {result}" })

if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_model()
	app.run()



app.run()