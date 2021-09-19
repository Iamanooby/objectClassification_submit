import itertools
import os
import cv2

import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

working_path_dir = os.getcwd()

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

saved_model_path = working_path_dir+ '/assets'
print(saved_model_path)
print(os.listdir(saved_model_path))

#imported = tf.saved_model.load(saved_model_path)#tf loading
imported = tf.keras.models.load_model(saved_model_path )#keras loading

def prediction(local_image_path,save_image_bool):#save_image_bool determines if a file was passed (there was a saved image) or an image was passed
    image = np.zeros((512,512))

    if save_image_bool:
        image_path = working_path_dir + local_image_path
        print(image_path)
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(512,512))
    else:
        image = tf.keras.preprocessing.image.smart_resize(local_image_path, (512,512), interpolation='bilinear')

    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(
    1. / 255)

    image = normalization_layer(image)




    # x, y = next(iter(val_ds))
    # image = x[0, :, :, :]
    # true_index = np.argmax(y[0])
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()
    class_names = ('fabric', 'foliage', 'glass', 'leather', 'metal', 'paper', 'plastic', 'stone', 'water', 'wood')
    # Expand the validation image to (1, 224, 224, 3) before predicting the label
    prediction_scores = imported.predict(np.expand_dims(image, axis=0))
    predicted_index = np.argmax(prediction_scores)
    print("Predicted label: " + class_names[predicted_index])
    return class_names[predicted_index]

# prediction("/test_photos/shirt.jpg",False)