import keras
import tensorflow as tf
from keras.layers import Dense, Flatten,Activation,Dropout
from keras.applications import MobileNet,imagenet_utils
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers

base_model=MobileNet(weights='imagenet',include_top=False,input_shape=(224,224,3))


# Freeze the layers
base_model.trainable=False

# Add custom layers on top of the base model
model=models.sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128,activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5,activation='softmax')    
    
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit()
