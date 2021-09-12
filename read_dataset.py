import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import pathlib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential

train_dir = f'/media/kisna/nano_ti_data/DL_git/YOLO/dataset_fire_detection/Training'
val_dir = f'/media/kisna/nano_ti_data/DL_git/YOLO/dataset_fire_detection/Test'
train_batch_size = 32
ImgSize = 224
epochs = 2

print("Reading dataset.....")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=train_dir,
    labels="inferred",
    label_mode="int",   
    image_size=(ImgSize, ImgSize),
    color_mode="rgb",
    batch_size=train_batch_size,
    shuffle=True,
    seed=123
)
class_names = train_ds.class_names
print(class_names)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=val_dir,
    labels="inferred",
    label_mode="int",   
    image_size=(ImgSize, ImgSize),
    color_mode="rgb",
    batch_size=train_batch_size,
    shuffle=True,
    seed=123
)

class_names = val_ds.class_names
print(class_names)


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

plt.show()

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 1


from tensorflow.keras.applications import EfficientNetB1

inputs = layers.Input(shape=(ImgSize, ImgSize, 3))
model = EfficientNetB1(weights='imagenet', input_tensor=inputs, include_top=False)

#model.trainable = False

x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
x = layers.BatchNormalization()(x)
top_dropout_rate = 0.2
x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
outputs = layers.Dense(num_classes, name="pred")(x)

model = tf.keras.Model(inputs, outputs, name="EfficientNet")

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)


model.compile(
  optimizer=optimizer,
  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data= val_ds,
)