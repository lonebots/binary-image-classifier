import keras
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras import datasets, layers, models
from keras.preprocessing import image


st.title("CLASSIFIER")
st.header("IMAGE > SHIP OR TRUCK")

# loading the model
def load_model():
  model=tf.keras.models.load_model('binary_classifier_1.hdf5')       
  return model

model=load_model()

file = st.file_uploader("FILE UPLOAD", type=["jpg", "png","jpeg"])

def loading_and_predicting(pic, model):
        size = (32,32)    
        image = ImageOps.fit(pic, size, Image.ANTIALIAS)       
        image = np.asarray(image)                              
        img = image.astype(np.float32) / 255.0                 
        final_img = img[np.newaxis,...]                        
        prediction = model.predict(final_img)
        return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)                     
    predictions = loading_and_predicting(image, model)
    value = np.argmax(predictions)                             
    if value == 1:
        st.markdown("TRUCK IMAGE")
    elif value == 0:
        st.markdown("SHIP IMAGE")
    else:
        st.markdown("CANNOT CLASSIFY")