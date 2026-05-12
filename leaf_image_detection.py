import tensorflow as tf
from keras.utils import image_dataset_from_directory
from keras.applications import MobileNet , InceptionV3
from keras .layers import Dense, Flatten , Dropout,GlobalAveragePooling2D
from keras.models import Sequential
from keras.callbacks  import EarlyStopping
import matplotlib.pyplot as plt


basic_model=InceptionV3(input_shape=(256,256,3),include_top=False)

basic_model.trainable=False


basic =Sequential([
basic_model,

GlobalAveragePooling2D(),

Dense(128,activation='relu'),
Dropout(0.5),

Dense(64,activation='relu'),
Dropout(0.5),

Dense(2,activation='softmax')
    
    
])


# loading des données d'entrainenment
directory="RiceleafDataset/train"

trainsdata=image_dataset_from_directory(
    directory=directory,
    labels="inferred",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(256, 256),
    batch_size=32,
    label_mode='categorical'
)


validationdata=image_dataset_from_directory(
    directory=directory,
    labels="inferred",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(256, 256),
    batch_size=32,
    label_mode='categorical'
)

basic.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# // entrinement du modèl

basic.fit( trainsdata,
    batch_size=16,
   epochs=150,
    validation_data=validationdata,
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]  
)

basic.save("RiceLeafDetectionModel_ofIcnceptionV3.keras")

with open("RiceLeafDetectionModel_ofInceptionV3.keras", "rb") as f:
    model_data = f.read()
    
plt.plot(basic.history.history['accuracy'], label='accuracy')
plt.plot(basic.history.history['val_accuracy'], label='val_accuracy')
plt.plot(basic.history.history['loss'], label='loss')
plt.plot(basic.history.history['val_loss'], label='val_loss')
plt.legend()

plt.show()    

