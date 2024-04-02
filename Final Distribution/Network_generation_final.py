import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import string
from sklearn.utils import shuffle
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.models import Sequential

tf.config.experimental.set_visible_devices([], 'GPU')

#data prep
df = pd.read_csv('handwritten_data_785.csv')

df.shape

X_data = df.values[:,1:]
Y_data = df.values[:,0]

X_data = X_data.reshape(len(X_data), 28, 28)

nr_to_letter = {k:v.upper() for k,v in enumerate(list(string.ascii_lowercase))}

seed = 17

X_data, Y_data = shuffle(X_data, Y_data, random_state=seed)

trainDataLength = (int)(len(X_data) * 0.85)

X_train = X_data[0:trainDataLength]
X_test = X_data[trainDataLength:]

Y_train = Y_data[0:trainDataLength]
Y_test = Y_data[trainDataLength:]

# !--- hardware specific limitation of dataset dimensions (blows surface memory)
X_train = X_train[0:200000]
Y_train = Y_train[0:200000]

# ---!
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],
                          X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],
                        X_test.shape[2],1)

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train/= 255.0
X_test/=255.0

#build network
model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(26,activation='softmax'))

model.compile(optimizer='adam' , 
              loss ='categorical_crossentropy',
              metrics =['accuracy'])

#model training
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        print(logs.get('accuracy'))
        if(logs.get('accuracy')>=0.95):
            print("\nReached 95.0% accuracy so cancelling training!")
            self.model.stop_training = True
        
callbacks = myCallback()

history = model.fit(X_train,Y_train, epochs = 1, 
                    validation_data=(X_test,Y_test),
                    callbacks = [callbacks])

eval = model.evaluate(X_test, Y_test)

print("Error: {}".format(eval[0]))
print("Accuracy: {}".format(eval[1]))

model.summary()
model.save("ANN_OCR_V3.h5")