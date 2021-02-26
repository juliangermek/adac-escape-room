import streamlit as st
import numpy as np

st.header("ADAC Escape Room: Challenge 3")

import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import os
import pandas as pd


# def save_uploaded_file(file):
#   # with open(os.path.join("tempDir", file.name), "wb") as f:
#   with open(os.path.join("model", file.name), "wb") as f:
#     f.write(file.getbuffer())
#   return st.success("Die Datei wurde erfolgreich hochgeladen.")

# file = st.file_uploader("Upload file")

# if st.button('Upload file'):
#   if file is not None:
#     save_uploaded_file(file)


# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5', compile=False)
# Supply labels
labels = ["Vogsphere", "Worlorn", "Solaris", "Krypton", "Arda", "Space"]

def classify_image(image_path):
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(image_path)

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    # image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)

    # Get label for most probably class
    max_idx = np.argmax(prediction)
    predicted_class = labels[max_idx]

    return predicted_class

st.write(classify_image('krypton_19.jpg'))

# Classify all
def classify_all_images():
    rootdir = 'shuttle_pictures/'
    result_dict = {}

    for subdir, _, files in os.walk(rootdir):
        if subdir != rootdir:
            shuttle = subdir.split("/")[-1]

            classification_list = []
            for file in files:
                classification_list.append(classify_image(os.path.join(subdir, file)))
            result_dict[shuttle] = classification_list

    return result_dict

result_dict = classify_all_images()
# st.write(result_dict)

# Put into df
df = pd.DataFrame(result_dict)
df = df[['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10', 'w11', 'w12', 'w13', 'w14', 'w15', 'w16', 'w17', 'w18', 'w19', 'w20']].transpose()
df.columns = ["foto_1", "foto_2", "foto_3", "foto_4"]
st.write(df)

# Filter
def filter_shuttles(row, should_see_vogsphere, should_see_worlorn, should_see_solaris, should_see_krypton, should_see_arda):
    """
    Return True if shuttle can see exactly the planets speci
fied
    Args:
        Booleans
    """
    sees_vogsphere = "Vogsphere" in row.tolist()
    sees_worlorn = "Worlorn" in row.tolist()
    sees_solaris = "Solaris" in row.tolist()
    sees_krypton = "Krypton" in row.tolist()
    sees_arda = "Arda" in row.tolist()

    count_planets = should_see_vogsphere + should_see_worlorn + should_see_solaris + should_see_krypton + should_see_arda

    count_space = row.tolist().count("Space")

    return (sees_vogsphere == should_see_vogsphere) & (sees_worlorn == should_see_worlorn) & (sees_solaris == should_see_solaris) & (sees_krypton == should_see_krypton) & (sees_arda == should_see_arda) & (count_space + count_planets == 4)
