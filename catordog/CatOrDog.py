
# coding: utf-8

# In[1]:


import numpy as np
from tqdm import tqdm
import cv2

np.random.seed(1024)

n = 1000
X = np.zeros((n, 224, 224, 3), dtype=np.uint8)
y = np.zeros((n, 1), dtype=np.uint8)

for i in tqdm(range(n//2)):
    X[i] = cv2.resize(cv2.imread('train/cat.%d.jpg' % i),(224, 224))
    X[i+n//2] = cv2.resize(cv2.imread('train/dog.%d.jpg' % i),(224, 224))

y[n//2:] = 1


# In[2]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[3]:


"""
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
print (X_train.shape)
"""
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.layers import  GlobalAveragePooling2D, Dropout, Dense,Input
base_model = ResNet50(input_tensor=Input((224, 224, 3)), weights='imagenet', include_top=False)

for layers in base_model.layers:
    layers.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.25)(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(base_model.input, x)


model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[4]:


"""
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

model = Sequential()
#c1
model.add(Convolution2D(
    batch_input_shape=(None, 28, 28, 3),
    filters=32,
    kernel_size=1,
    strides=1,
    padding='same',     # Padding method
    data_format='channels_first'))

model.add(Activation('relu'))

model.add(MaxPooling2D(
    pool_size=1,
    strides=1,
    padding='same',    # Padding method
    data_format='channels_first',
))

model.add(Convolution2D(64, 1, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (64, 7, 7)
model.add(MaxPooling2D(1, 1, 'same', data_format='channels_first'))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(2))
model.add(Activation('softmax'))

adam = Adam(lr=1e-7)

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
"""


# In[8]:


model.fit(X_train, y_train, batch_size=64, epochs=5, validation_data=(X_test, y_test))
model.save('train_result_model.h5')

