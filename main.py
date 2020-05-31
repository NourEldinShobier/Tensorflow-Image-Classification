import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import datasets, layers, models
from utils import plot_value_array, plot_image, verify_prediction

'''
load dataset
'''

# 60,000 images for training the network and 10,000 images to evaluate accurately
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# assign a label to each image class
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

'''
explore data
'''

print(f'train images (shape): {train_images.shape}')
print(f'train labels length: {len(train_labels)}')
print(f'test labels length: {len(test_labels)}')
print(f'train labels: {train_labels}')

'''
preprocess dataset
'''

# display pixels values (0 to 255) (grayscale)
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# scale pixels values to a range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# display 25 picture to verify grayscale scaling
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(classes[train_labels[i]])
plt.show()

'''
init the model
'''

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # 1st layer: flatten 28 x 28 pixels to 784 nodes/neurons
    layers.Dense(128, activation='relu'),  # 2nd layer: use relu as activation function
    layers.Dense(10)
])

print(model.summary())

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train the model
history = model.fit(train_images, train_labels, epochs=10)

# test the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)

'''
display accuracy and loss during training time
'''

acc = history.history['accuracy']
loss = history.history['loss']

epochs_range = range(10)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.legend(loc='lower right')
plt.title('Training Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.legend(loc='upper right')
plt.title('Training Loss')
plt.show()

# --------

'''
make prediction
'''

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

print(f'predictions: {predictions[0]}')

# print label has the highest confidence value
print(f'max confidence value: {np.argmax(predictions[0])}')

'''
verify predictions visually
'''

verify_prediction(0,
                  predictions,
                  test_labels,
                  test_images,
                  classes)

verify_prediction(12,
                  predictions,
                  test_labels,
                  test_images,
                  classes)

# plot random images with their predictions
num_rows = 5
num_cols = 3

num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images, classes)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# grab random image from dataset
img = test_images[1]

# add the image to a batch where it's the only member
img = (np.expand_dims(img, 0))

print(f'image shape: {img.shape}')

predictions_single = probability_model.predict(img)

print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), classes, rotation=45)

print(f'image class: {print(np.argmax(predictions_single[0]))}')
