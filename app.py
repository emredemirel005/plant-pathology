import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from .classify_image import img_classification



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
        
        result = img_classification(image_or,model)
        result = class_names[np.argmax(result)]
        image_result = plt.imshow(image_or)
        plt.title(result)
        st.pyplot()