import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
import keras

train_path = "/Users/theodatta/Downloads/mathsymbols_data/"

# With datagen, I have the option to preprocess or change image dimensions
train_data = ImageDataGenerator(
    validation_split = 0.25
)

train = train_data.flow_from_directory(
    train_path, 
    target_size = (45, 45), 
    color_mode = 'grayscale',
    class_mode = 'categorical',
    subset='training') 

val = train_data.flow_from_directory(
    train_path, 
    target_size = (45, 45), 
    color_mode = 'grayscale',
    class_mode = 'categorical',
    subset='validation') 


# model = keras.models.Sequential()
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(100, activation='relu', kernel_initializer='random_normal', input_dim = 2025))
# model.add(keras.layers.Dense(100, activation='relu', kernel_initializer='random_normal', input_dim=100))
# model.add(keras.layers.Dense(82, activation='softmax', kernel_initializer='random_normal', input_dim=100))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(45, 45, 1)))
model.add(tf.keras.layers.MaxPool2D(strides=2))
model.add(tf.keras.layers.Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(strides=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1000, activation='relu'))
model.add(tf.keras.layers.Dense(200, activation='relu'))
model.add(tf.keras.layers.Dense(82, activation='softmax'))

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01),loss=keras.losses.MeanSquaredError(), metrics = ['accuracy'])
# Different params:
# Optimizers: SGD, Adam, RMSProp
# Losses: keras.losses.MeanSquaredError()

#Interesting for future models:
# model.lr_find()
# model.recorder.plot(suggestion = True)

model.fit(train, validation_data=val, epochs=25, verbose=1, batch_size=256,
        callbacks = keras.callbacks.EarlyStopping(monitor ="val_loss", mode ="min", patience = 5, restore_best_weights = True))


# classes = ['}','{',']','[','z','y','X','w','v','u','times','theta','tan','T','sum','sqrt','sin','sigma',
#                'S','rightarrow','R','q','prime','pm','pi','phi','p','o','neq','N','mu','M','lt','log','lim','leq',
#                'ldots','lambda','l','k','j','int','infty','in','i','H','gt','geq','gamma','G','forward_slash','forall','f',
#                'f','exists','e','div','Delta','d','cos','C','beta','b','ascii_124','alpha','A','=','9','8','7','6',
#                '5','4','3','2','1','0','-',',','+','(',')','!'],
