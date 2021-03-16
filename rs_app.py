import tensorflow as tf
model = tf.keras.models.load_model('model.hdf5')


import streamlit as st
st.write("""
# Check Dam and Farm Pond prediction
"""
)
st.write("This is a simple image classification web app to predict check dam or farm pond")
file = st.file_uploader("Please upload an image file", type=["jpg"])

import cv2
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):
    size = (150,150)
    image = ImageOps.fit(image_data, size)
    image = np.asarray(image)
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = (cv2.resize(image, dsize=(150, 150)))/255.
# , interpolation=cv2.INTER_CUBIC   , Image.ANTIALIAS
    img_reshape = img_resize[np.newaxis,...]

    prediction = model.predict(img_reshape)

    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
st.image(image, use_column_width=True)
prediction = import_and_predict(image, model)

if(prediction[0][0]>0.5):
    st.write("Its a Farm pond -> ",prediction[0][0]*100)

else:
    st.write("Its a Check dam",100-prediction[0][0]*100)


st.text("Probability (0: Check dam, 1: Farm pond)")
st.write(prediction[0][0])