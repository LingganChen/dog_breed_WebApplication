import tensorflow as tf
import math
import requests
import numpy as np
#import matplotlib.pyplot as plt
import PIL
from PIL import Image
from io import BytesIO
from tensorflow import keras
from keras import layers
from sklearn.metrics import roc_curve
from keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array
from keras.applications import InceptionV3
from keras.models import load_model
from keras.models import model_from_json
from keras.layers import GlobalAveragePooling2D, Dropout, Dense

img_h = img_w = 224
dims = (224,224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale = 1.0/255,
    #data augmentation
    rotation_range = 15,
    zoom_range = (0.95, 0.95),
    horizontal_flip = True,
    vertical_flip = True
)

val_datagen = ImageDataGenerator(rescale = 1.0/255)

test_datagen = ImageDataGenerator(rescale = 1.0/255)

train = train_datagen.flow_from_directory(
    "train/",
    target_size = dims,
    batch_size = batch_size,
    color_mode = 'rgb',
    class_mode = 'categorical',
    seed = 123
)

val = val_datagen.flow_from_directory(
    "valid/",
    target_size = dims,
    batch_size = batch_size,
    color_mode = 'rgb',
    class_mode = 'categorical',
    seed = 123
)

test = test_datagen.flow_from_directory(
    "test/",
    target_size = dims,
    batch_size = batch_size,
    color_mode = 'rgb',
    class_mode = 'categorical',
    seed = 123
)

# labels
label_mapper = np.asarray(list(train.class_indices.keys()))

IMG_SHAPE = (224,224,3)
v3model = InceptionV3(input_shape = IMG_SHAPE, include_top=False, weights='imagenet', classes=70)
# freezes the v3 model's weights
v3model.trainable = False
# create new model 
model = keras.Sequential()
# add v3model layer
model.add(v3model)
model.add(Dropout(0.2))
#reduces the spatial dimensions of the input to a single value per channel
model.add(GlobalAveragePooling2D())
model.add(keras.layers.Dense(70,activation='softmax'))

# callbacks
# reduce lr if val_loss not improving in 3 epochs
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=2, factor=0.001)
# save best weight
checkpoint_filepath = 'checkpoint/'
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)


model.compile(
    optimizer=keras.optimizers.Adam(0.0001),
    loss = keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"])

#load best weights
model.load_weights(checkpoint_filepath)

#train model
"""model.fit(
    train, 
    validation_data = val,
    steps_per_epoch = train.samples//batch_size,
    validation_steps = val.samples//batch_size,
    epochs=10,
    verbose=1,
    callbacks=[reduce_lr, cp_callback]
    )"""


model.evaluate(test)

# write model to json format
model_json = model.to_json()
with open("ImageClassifier.json", "w") as json_file:
    json_file.write(model_json)

# save weights to HDF5
model.save_weights("ImageClassifier.h5")