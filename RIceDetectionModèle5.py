import keras
from keras.layers import Dense, Flatten,Activation,Dropout
from keras.applications import EfficientNetV2B2,ResNet152V2,ResNet101
from keras.metrics import categorical_crossentropy
# from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers


base_model=ResNet101(
    input_shape=(256,256,3),
    include_top=False 
    )

# permet dd geler le corps c'est a dire le corp de modèle pour qu'il utilise ces connaissances apprise sur les données d'entrainement 
base_model.trainable=False

model=models.Sequential([
    
     layers.Input(shape=(256, 256, 3)), # Le modèle accepte du 256
    layers.Resizing(256, 256),
    base_model,
    # le second de layers.flatten() il est là pour applatire les feature maps du modèle en vecteur c'est a dire une liste de nombre.
    # Au fait ce qu'il fait est qu'il prend chaque matrice du feauture et faire sa moyenne pour obtenir un nombre unique pour chaque  matrice.
    layers.GlobalAveragePooling2D(),
    #  Ce sont les couches de classification que nous allons entrainer pour notre tâche de classification de 6 classes de riz
    layers.Dense(128,activation='relu'),
    # Pour empêche le surapprentissage on utilise la technique de dropout qui consiste à désactiver aléatoirement des neurones pendant l'entrainement pour éviter que 
    # le modèle ne s'adapte trop aux données d'entrainement et qu'il puisse généraliser mieux sur les données de test.
    layers.Dropout(0.5),
    layers.Dense(4,activation='softmax') 
       
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
 
Train_label_dir="dataset/RiceDiseaseDataset/train"

trainds=keras.utils.image_dataset_from_directory(
    directory=Train_label_dir,
  labels = "inferred",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(256, 256),
  batch_size=32,
  label_mode='categorical',
)
          
validationds=keras.utils.image_dataset_from_directory(
 directory=Train_label_dir,
  labels = "inferred",
  subset="validation",
  image_size=(256, 256),
  validation_split=0.2,
  batch_size=32,
  seed=123,
  label_mode='categorical',
)

# pour savoir quand s'arreter l'entrainement pour éviter le surapprentissage 

from keras.callbacks import EarlyStopping

early_stopping= EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# entrainement du model
model.fit(
trainds,
validation_data = validationds,
epochs=180
,
 callbacks=[early_stopping]  
)

model.save("NewModel_ofResNet101.keras")

import matplotlib.pyplot as plt
plt.plot(model.history.history['accuracy'], label='accuracy')
plt.plot(model.history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()



