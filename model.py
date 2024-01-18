import IPython.display as display
import numpy as np
import pandas as pd
from PIL import Image
#from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import model_from_json


# Load model
model_json = open("ImageClassifier.json",'r')
loaded_model_json = model_json.read()
model_json.close()
model = model_from_json(loaded_model_json)
model.load_weights("ImageClassifier.h5")

# Preparing and pre-processing the image
def preprocess_img(img_path):
    op_img = Image.open(img_path)
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, 224, 224, 3)
    return img_reshape

def predict_result(image):

    # Load the image
    """img = load_img(image_path, target_size=(224, 224))  # Adjust target_size as needed
    display.display(img)
    # Convert the image to a NumPy array
    arr = img_to_array(img)
    arr = arr / 255.0
    arr = np.expand_dims(arr, 0)"""
    # read csv
    df = pd.read_csv("dogs.csv")
    # Extract unique labels from the "label" column and convert them to a Python list
    label_mapper= df["labels"].unique().tolist()
    # Make predictions using the model
    predictions = model.predict(image)
    #print(predictions.shape)

    # Get the index with the highest probability
    idx = np.argmax(predictions)

    # Return the predicted label and probability
    return label_mapper[idx] #, predictions[0][idx]
