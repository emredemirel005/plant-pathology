import cv2
import numpy as np
from matplotlib import pyplot as plt

def img_classification(img,model):
    image = np.asarray(img)
    image = cv2.resize(image,(224,224))
    image = image/255
    image = np.expand_dims(image,0)

    result = model.predict(image)
    return result