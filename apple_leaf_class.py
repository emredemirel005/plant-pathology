import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

model = load_model('best_weights.h5')
IMG_SIZE = 224

def check_apple_leaf(image_dir):
    image_result = Image.open(image_dir)
    
    test_image = image.load_img(image_dir,target_size=(IMG_SIZE,IMG_SIZE))
    test_image = image.img_to_array(test_image)
    test_image = test_image/255
    test_image = np.expand_dims(test_image,axis=0)
    result = model.predict(test_image)

    Categories = ['healthy','multiple_disease','rust','scab']
    image_result = plt.imshow(image_result)
    plt.title(Categories[np.argmax(result)])
    plt.show()