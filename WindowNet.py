import tensorflow as tf
import cv2
import imghdr
import os
from matplotlib import pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.metrics import Precision, Recall, BinaryAccuracy


data_dir = 'data'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

img = cv2.imread(os.path.join('data', 'windowed', 'image (2).jpeg'))

#remove images without certain file types
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('issue with image {}'.format(image_path))
            #os.remove(image_path)

#import image data
data = tf.keras.utils.image_dataset_from_directory('data')

data = data.map(lambda x,y: (x/255, y))

data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

scaled_iterator = data.as_numpy_iterator()

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1) + 1


train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

#create model, instance of sequential class
model = Sequential()

#adds convolutional layer and pooling layer
#conv layer has 16 filters, 3x3px dimensions, 1px step
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D()) #condensing layer

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten()) #flattens convolutional layers for input to dense layer

#fully connected dense layer
model.add(Dense(256, activation='relu'))

#output layer
model.add(Dense(1, activation='sigmoid'))

#compile model, adam optimizer,
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

#return summary to terminal
model.summary()

logdir = 'logs'
#collect training history
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

#fit model to training data, log training history w/ callbacks
hist = model.fit(train, epochs = 20, validation_data=val, callbacks=[tensorboard_callback])

#plot loss
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize = 20)
plt.legend(loc='upper left')
plt.show()

#plot accuracy
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize = 20)
plt.legend(loc='upper left')
plt.show()


precision = Precision()
recall = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    precision.update_state(y, yhat)
    recall.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f'Precsion:{precision.result()}, Recall:{recall.result()}, Accuracy:{acc.result()}')

