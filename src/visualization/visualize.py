from pathlib import Path
import streamlit as st
from PIL import Image
import hashlib
import pandas as pd
import tensorflow as tf
import io
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas


@st.experimental_memo
def get_model():
    return tf.keras.models.load_model("models/model.h5")


def perform_ocr(image):
    model = get_model()

    img = Image.fromarray(image)
    img = img.convert("L")
    img = img.resize((28, 28))
    arr = np.asarray(img).astype(np.float32).reshape((1, 28, 28))
    arr = 255 - arr
    arr = arr / 255
    predictions = model.predict(arr, verbose=False)[0]
    prediction = np.argmax(predictions)
    return prediction, predictions


# print("render")

st.write("# MNIST OCR")

col1, col2 = st.columns(2)
with col1:
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=4,
        stroke_color="black",
        background_color="#fff",
        update_streamlit=True,
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        if len(canvas_result.json_data["objects"]) > 0:
            # st.image(canvas_result.image_data)
            prediction, predictions = perform_ocr(canvas_result.image_data)
            with col2:
                st.metric("Prediction", prediction)
                st.bar_chart(
                    pd.DataFrame({"index": range(0, 10), "predictions": predictions}).set_index(
                        "index"
                    ),
                    height=200,
                )
