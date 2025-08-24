import streamlit as st
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from scipy.spatial.distance import cosine
import joblib

# Load model and features once (cached for performance)
@st.cache_resource
def load_cnn_model():
    return load_model("fashion_model.keras")

@st.cache_data
def load_features():
    return joblib.load("fashion_features.pkl")

# Image preprocessing
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

# Feature extraction
def extract_features(model, preprocessed_img):
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features

# Recommendation logic
def recommend_fashion_items_cnn(uploaded_image_path, all_features, all_image_names, model, top_n=5):
    preprocessed_img = preprocess_image(uploaded_image_path)
    input_features = extract_features(model, preprocessed_img)

    similarities = [1 - cosine(input_features, other_feat) for other_feat in all_features]

    input_image_name = os.path.basename(uploaded_image_path)
    similar_indices = [
        idx for idx in np.argsort(similarities)[-top_n-1:]
        if os.path.basename(all_image_names[idx]) != input_image_name
    ]

    recommended_images = [all_image_names[idx] for idx in similar_indices[:top_n]]
    return recommended_images


st.set_page_config(page_title="Fashion Recommender", layout="wide")
st.title("IFRA - Intelligent Fashion Recommender")

# Sidebar

st.sidebar.image(r"C:\Users\Shuhaib\Downloads\assets_task_01jvqc3g2xe4at1emfrbmx4cjc_1747762477_img_0.jpg", width=100)
st.sidebar.header(" Upload ")

uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Slider for number of recommendations
top_n = st.sidebar.slider("Number of recommendations", min_value=1, max_value=10, value=5)

# Load model and features
model = load_cnn_model()
all_features, all_image_names = load_features()

# Main Content
if uploaded_file:
    with st.spinner("🔍 Finding similar fashion items..."):
        # Save uploaded file temporarily
        temp_path = os.path.join("temp_uploaded_image.jpg")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Display uploaded image
        st.subheader("📤 Uploaded Image")
        st.image(temp_path, caption="Uploaded Image", width=300)

        # Get recommendations
        recommendations = recommend_fashion_items_cnn(temp_path, all_features, all_image_names, model, top_n=top_n)

        # Display recommended images
        st.subheader("🛍️ You may also like:")
        cols = st.columns(len(recommendations))
        for col, img_path in zip(cols, recommendations):
            img_full_path = os.path.join(
                r"C:\Users\Shuhaib\Downloads\shuaib\Project\DATA_S~1\FASHIO~1\WOMEN-~1\WOMENF~1",
                os.path.basename(img_path)
            )
            try:
                col.image(img_full_path, use_column_width=True, caption=os.path.basename(img_path))
            except Exception as e:
                col.error(f"Image not found: {os.path.basename(img_path)}")
else:
    st.info("👈 Please upload an image from the sidebar to get started.")