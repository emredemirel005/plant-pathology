import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

st.set_page_config(page_title="Plant Pathalogy", page_icon=":wave:")
st.set_option('deprecation.showPyplotGlobalUse', False)

model = load_model("./best_weights.h5")
class_names = ["healthy","multiple_diseases","rust","scab"]

st.title("Plant Disease Detection")
st.markdown("Upload an image of the plant leaf")

plant_image = st.file_uploader("Choose an image...")
submit = st.button("Predict")

if submit:
    if plant_image is not None:
        # convert file to an cv2 image
        
        image_or = Image.open(plant_image)

        #display image
        #st.image(image_or,channels="RGB",width=256)
        
        image = np.asarray(image_or)
        image = image_or.resize((224,224))
        #image = image/255
        image = np.expand_dims(image,0)

        result = model.predict(image)
        result = class_names[np.argmax(result)]
        image_result = plt.imshow(image_or)
        plt.title(result)
        st.pyplot()
