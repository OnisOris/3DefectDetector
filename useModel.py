import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing import image

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

model = tf.keras.models.load_model('my_model.keras')
batch_size = 32

path1 = './Defects/defected/bed_not_stick_55.jpg'
path2 = './Defects/no_defected/scratch_55.jpg'
path3 = 'img/6-maxresdefault_jpg.rf.7557ffe7ef52e71e5596c55d25ce346e.jpg'
path4 = 'img/25-tumblr_mu5j9vp8q71sukl1qo1_1280_jpg.rf.56a09eec3bbdeb34d405f146d91b3899.jpg'

path = [path1, path2, path3, path4]

for item in path:
    img_width, img_height = 180, 180
    img = image.load_img(item, target_size=(img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    print(prediction)
