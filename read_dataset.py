import tensorflow as tf

train_dir = f'/media/kisna/nano_ti_data/DL_git/YOLO/dataset_fire_detection/Training'
test_dir = f'/media/kisna/nano_ti_data/DL_git/YOLO/dataset_fire_detection/Test'
train_batch_size = 32
ImgSize = 224
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

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=test_dir,
    labels="inferred",
    label_mode="int",   
    image_size=(ImgSize, ImgSize),
    color_mode="rgb",
    batch_size=train_batch_size,
    shuffle=True,
    seed=123
)

class_names = test_ds.class_names
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