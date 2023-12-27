#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import math
import requests
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array
from tensorflow.keras.applications import InceptionResNetV2,InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense


# In[8]:


img_h = img_w = 224
dims = (224,224)
batch_size = 32


# In[11]:


train_datagen = ImageDataGenerator(
    rescale = 1.0/255,
    #data augmentation
    rotation_range = 15,
    zoom_range = (0.95, 0.95),
    horizontal_flip = True,
    vertical_flip = True
)


# In[12]:


val_datagen = ImageDataGenerator(rescale = 1.0/255)


# In[13]:


test_datagen = ImageDataGenerator(rescale = 1.0/255)


# In[14]:


train = train_datagen.flow_from_directory(
    "train/",
    target_size = dims,
    batch_size = batch_size,
    color_mode = 'rgb',
    class_mode = 'categorical',
    seed = 123
)


# In[15]:


val = val_datagen.flow_from_directory(
    "valid/",
    target_size = dims,
    batch_size = batch_size,
    color_mode = 'rgb',
    class_mode = 'categorical',
    seed = 123
)


# In[16]:


test = test_datagen.flow_from_directory(
    "test/",
    target_size = dims,
    batch_size = batch_size,
    color_mode = 'rgb',
    class_mode = 'categorical',
    seed = 123
)


# In[17]:


label_mapper = np.asarray(list(train.class_indices.keys()))
label_mapper


# In[2]:


IMG_SHAPE = (224,224,3)
v3model = InceptionV3(input_shape = IMG_SHAPE, include_top=False, weights='imagenet', classes=70)
#freezes the v3 model's weights
v3model.trainable = False


# In[159]:


v3model.summary()


# In[3]:


model = keras.Sequential()
model.add(v3model)


# In[4]:


model.add(Dropout(0.2))
#reduces the spatial dimensions of the input to a single value per channel
model.add(GlobalAveragePooling2D())
model.add(keras.layers.Dense(70,activation='softmax'))


# In[162]:


model.summary()


# In[5]:


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

#load best weights
model.load_weights(checkpoint_filepath)


# In[164]:


model.compile(
    optimizer=keras.optimizers.Adam(0.0001),
    loss = keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"])


# In[165]:


logs = model.fit(
    train, 
    validation_data = val,
    steps_per_epoch = train.samples//batch_size,
    validation_steps = val.samples//batch_size,
    epochs=10,
    verbose=1,
    callbacks=[reduce_lr, cp_callback]
    )

#load best weights
model.load_weights(checkpoint_filepath)


# In[166]:


model.evaluate(test)


# In[167]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Model accuracy
plt.plot(logs.history['accuracy'])
plt.plot(logs.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# In[168]:


# Model Losss
plt.plot(logs.history['loss'])
plt.plot(logs.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# In[172]:


# full path for saving the model
save_dir = "saved/"

# Ensure the directory exists (create it if it doesn't)
os.makedirs(save_dir, exist_ok=True)

# Save the model in the TensorFlow SavedModel format
model.save(os.path.join(save_dir, 'model'), save_format='tf')


# In[19]:


import IPython.display as display

def predictor(image_path):
    # Load the image
    img = load_img(image_path, target_size=(224, 224))  # Adjust target_size as needed
    display.display(img)

    # Convert the image to a NumPy array
    arr = img_to_array(img)
    arr = arr / 255.0
    arr = np.expand_dims(arr, 0)

    # Make predictions using the model
    predictions = model.predict(arr)
    print(predictions.shape)

    # Get the index with the highest probability
    idx = np.argmax(predictions)

    # Return the predicted label and probability
    return label_mapper[idx], predictions[0][idx]


# In[21]:


result = predictor('daidai.jpg')
print(result)


# In[ ]:




