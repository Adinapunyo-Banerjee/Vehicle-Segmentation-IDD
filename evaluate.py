# Timing
from time import time
# Common
import os
from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
# Data
import cv2 as cv
import tensorflow.image as tfi
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import keras

model = keras.models.load_model("./MODELS/idd_unet_model_na.h5")

class_map_df = pd.read_csv("./class_dict.csv")
class_map = []
for index,item in class_map_df.iterrows():
    class_map.append(np.array([item['r'], item['g'], item['b']]))

def load_image(path, SIZE):
    img = load_img(path)
    img = img_to_array(img)
    img = tfi.resize(img, (SIZE, SIZE))
    img = tf.cast(img, tf.float32)
    img = img/255.
    return img

def show_image(image, title=None, cmap=None, alpha=1):
    plt.imshow(image, cmap=cmap, alpha=alpha, vmin=0, vmax=255)
    if title is not None:
        plt.title(title)
    plt.axis('off')

SIZE = 512
V = 30

root_path = './'
test_path = './test/'
save_path = './Predictions/'

test_image_paths = sorted(glob(root_path + 'test/' + "*.png"))

file_names = []
for root, directory, file in os.walk(test_path):
    file_names.append(file)
file_names = file_names[0]
print(file_names)

###########################################################################

def image_class_map(y_pred, name):
    image = np.zeros((y_pred.shape[1], y_pred.shape[2]), dtype=np.int32)
    for j in range(y_pred.shape[1]):
        for k in range(y_pred.shape[2]):
            if y_pred[0][j][k] == 7:
                image[j][k] = 0
                continue
            image[j][k] = (y_pred[0][j][k]+1)*V

    # show_image(image, cmap = 'gray')
    # plt.show()

    cv.imwrite("./Predictions/{}.png".format(name), image)

###########################################################################

N = 10                          # Max number of predictions

time_map = []

for i, path in tqdm(enumerate(test_image_paths), desc="Test Images"):
    if i > N:
        break

    image = load_image(path, SIZE=SIZE)
    image = np.expand_dims(image, axis=0)

    t1 = time()
    pred1 = model.predict(image)
    time_map.append(time()-t1)

    pred1_f = np.argmax(pred1, axis=3)
    image_class_map(pred1_f, file_names[i])

# Prediction time log

tf = pd.DataFrame(columns=['image_name', 'prediction_time'])

for i in range(N):
    tf.loc[len(tf.index)] = [file_names[i], time_map[i]]

tf.to_csv("time_stamps.csv")