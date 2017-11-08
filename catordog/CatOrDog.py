from PIL import Image

import os
#os.chdir('./train')

#img=Image.open('cat.0.jpg')
#print (img.format, img.size, img.mode)#img.shape是不行的，shape针对的是array格式
#img.show()
"""
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = load_img('cat.20.jpg')  # this is a PIL image
print (img.format,img.size,img.mode)

x = img_to_array(img)# this is a Numpy array with shape (3, 150, 150)
print (x.shape)
x=x.reshape((1,)+x.shape)
print (x.shape)
i=0
for batch in datagen.flow(x,batch_size=1,save_to_dir='preview',save_prefix='cat', save_format='jpeg'):
    i+=1
    if i>20:
        break
"""
import numpy as np
from tqdm import tqdm
import cv2

np.random.seed(2017)

n = 25000
X = np.zeros((n, 224, 224, 3), dtype=np.uint8)
y = np.zeros((n, 1), dtype=np.uint8)

for i in tqdm(range(n/2)):
    X[i] = cv2.resize(cv2.imread('train/cat.%d.jpg' % i), (224, 224))
    X[i+n/2] = cv2.resize(cv2.imread('train/dog.%d.jpg' % i), (224, 224))

y[n/2:] = 1
