from pathlib import Path
import streamlit as st
from io import StringIO

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

current_dir=Path(__file__).parent if "__file__" in locals() else Path.cwd()

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

PAGE_TITLE = "Plant Pathalogy"
PAGE_ICON = ":wave:"

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)

upload_file =st.file_uploader(label='load your plant leaf file')

if upload_file is not None:

    # to convert to as str based IO
    stringio = StringIO(str(upload_file))

    # to read file as str
    string_data = stringio.read()

    img = Image.open(string_data)
    st.image(img,width=224,height=224)

