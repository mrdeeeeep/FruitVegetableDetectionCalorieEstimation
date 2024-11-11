import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt



calorie = [52, 89, 43, 31, 25, 20, 41, 25, 40, 86, 15, 25, 149, 80, 69, 29, 61, 29, 14, 60, 40, 47, 31, 57, 81, 50, 83, 77, 16, 147, 23, 86, 86, 18, 28, 30]



test_set = tf.keras.utils.image_dataset_from_directory(
    'test',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

cnn = tf.keras.models.load_model('trained_model5.h5')

#Test Image Visualization
import cv2
image_path = 'mockup-graphics-G693X4i3F2I-unsplash.jpg'
# Reading an image in default mode
img = cv2.imread(image_path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #Converting BGR to RGB

image = tf.keras.preprocessing.image.load_img(image_path,target_size=(64,64))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = cnn.predict(input_arr)

print(predictions)

result_index = np.argmax(predictions) #Return index of max element
print(result_index)

#Single image Prediction
print("It's a {}".format(test_set.class_names[result_index]))
print(calorie[result_index])