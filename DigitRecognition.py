import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import cv2


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

x_train = x_train.reshape(-1, 784).astype("float32")
x_test = x_test.reshape(-1, 784).astype("float32")

scaled_x = StandardScaler()

x_train_scaled = scaled_x.fit_transform(x_train)
x_test_scaled = scaled_x.transform(x_test)
'''
model = keras.Sequential(
    [
        tf.keras.Input(shape=(784,)),
        layers.Dense(units=128, activation='relu'),
        layers.Dense(units=128, activation='relu'),
        layers.Dense(units=10, activation='linear')
    ]
)

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(0.001),
              metrics=['accuracy'],
              )
model.fit(x_train_scaled, y_train, epochs=5)

model.save('DigitRecognition.keras')
'''
'''
model = tf.keras.models.load_model('DigitRecognition.keras')
loss, accuracy = model.evaluate(x_test_scaled,y_test)
print(loss)
print(accuracy)
'''
model = tf.keras.models.load_model('DigitRecognition.keras')

image_num = 1
while os.path.isfile(f"Digits/Digit_{image_num}.png"):
    try:
        image = cv2.imread(f"Digits/Digit_{image_num}.png")[:,:,0]
        image = np.invert(np.array(image))
        image = np.reshape(image, (-1, 784))
        prediction = model.predict(image)
        print(f"The digit is {np.argmax(prediction)}")
        plt.imshow(image.reshape(28, 28), cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        print("Error", e)
    finally:
        image_num += 1



