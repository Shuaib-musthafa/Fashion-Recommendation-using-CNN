# Fashion Recommendation System using Image Features

## Overview
This project implements a Fashion Recommendation System that leverages deep learning and image processing to recommend visually similar fashion items. Using the VGG16 Convolutional Neural Network (CNN) model pre-trained on ImageNet, the system extracts image features from a dataset of fashion images and ranks them based on visual similarity to a user-provided input image.

## Key Features
  1.Dataset Processing: Extracts and processes a large dataset of fashion images to prepare them for feature extraction.
  2.CNN Feature Extraction: Utilizes the VGG16 model to extract meaningful image features for comparison.
  3.Cosine Similarity Ranking: Measures similarity between feature vectors to find the most visually similar fashion items.
  4.Recommendations Visualization: Displays the input image alongside its top recommendations.
## Requirements
  Python 3.x
  TensorFlow
  Keras
  NumPy
  Matplotlib
  PIL (Pillow)
  SciPy
  
## Install the necessary libraries using the command:

bash
  pip install tensorflow keras numpy matplotlib pillow scipy
## Dataset
  You will need a dataset of fashion images for this project. Ensure your dataset includes a wide variety of fashion items (e.g., dresses, shirts, pants, etc.) with diverse colors, patterns, and styles.

In this example, we use a zip file containing a directory of images named women-fashion.zip. Ensure the dataset follows a consistent file format (e.g., .jpg, .png, .jpeg, .webp).

## Project Structure
  1.Extract Dataset: The dataset is extracted from a zip file and processed for feature extraction.
  2.Feature Extraction: Uses VGG16 to extract feature vectors from images.
  3.Similarity Measurement: Calculates cosine similarity between feature vectors to rank fashion items based on visual similarity.
  4.Recommendation System: Recommends the top N visually similar items for a given input image.
## Setup and Execution
  1. Extract the Dataset
  The script below extracts images from a zip file:

  python
    
    from zipfile import ZipFile
    import os

    zip_file_path = "path_to_your_zip_file"
    extraction_directory = "path_to_extraction_directory"

    if not os.path.exists(extraction_directory):
    os.makedirs(extraction_directory)

    with ZipFile(zip_file_path, 'r') as zip_ref:
      zip_ref.extractall(extraction_directory)

    extracted_files = os.listdir(extraction_directory)
    print(extracted_files[:10])
2. Preprocess the Images
  To display the first image from the dataset, you can use the following code:

  python
   
    from PIL import Image
    import matplotlib.pyplot as plt

    def display_image(file_path):
      image = Image.open(file_path)
      plt.imshow(image)
      plt.axis('off')
      plt.show()

    # Display the first image
    first_image_path = "path_to_first_image"
    display_image(first_image_path)
3. Load and Preprocess the Images for VGG16
  Generate a list of image file paths and preprocess each image to prepare for feature extraction:

  python
    
    import glob
    image_directory = 'path_to_images_directory'

    image_paths_list = [file for file in glob.glob(os.path.join(image_directory, '*.*')) if file.endswith(('.jpg', '.png', '.jpeg', 'webp'))]
    print(image_paths_list)
4. Extract Features from Images
  Use the pre-trained VGG16 model to extract image features:

  python
    
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    from tensorflow.keras.models import Model
    import numpy as np

    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.output)

    def preprocess_image(img_path):
      img = image.load_img(img_path, target_size=(224, 224))
      img_array = image.img_to_array(img)
      img_array_expanded = np.expand_dims(img_array, axis=0)
      return preprocess_input(img_array_expanded)

    def extract_features(model, preprocessed_img):
      features = model.predict(preprocessed_img)
      flattened_features = features.flatten()
      normalized_features = flattened_features / np.linalg.norm(flattened_features)
      return normalized_features

    all_features = []
    all_image_names = []

    for img_path in image_paths_list:
      preprocessed_img = preprocess_image(img_path)
      features = extract_features(model, preprocessed_img)
      all_features.append(features)
      all_image_names.append(os.path.basename(img_path))
5. Fashion Recommendation Function
  Implement the recommendation function based on cosine similarity:

python
 
    from scipy.spatial.distance import cosine
    import matplotlib.pyplot as plt
    from PIL import Image

    def recommend_fashion_items_cnn(input_image_path, all_features, all_image_names, model, top_n=5):
      preprocessed_img = preprocess_image(input_image_path)
      input_features = extract_features(model, preprocessed_img)
      
      similarities = [1 - cosine(input_features, other_feature) for other_feature in all_features]
      input_image_name = os.path.basename(input_image_path)
      similar_indices = [idx for idx in np.argsort(similarities)[-top_n-1:] if all_image_names[idx] != input_image_name]

      plt.figure(figsize=(15, 10))
      plt.subplot(1, top_n + 1, 1)
      plt.imshow(Image.open(input_image_path))
      plt.title("Input Image")
      plt.axis('off')

      for i, idx in enumerate(similar_indices[:top_n], start=1):
          image_path = os.path.join(image_directory, all_image_names[idx])
          plt.subplot(1, top_n + 1, i + 1)
          plt.imshow(Image.open(image_path))
          plt.title(f"Recommendation {i}")
          plt.axis('off')
  
          plt.tight_layout()
          plt.show()
6. Test the Recommendation System
Provide the path to the input image and visualize recommendations:

  python
   
    input_image_path = "path_to_input_image"
    recommend_fashion_items_cnn(input_image_path, all_features, all_image_names, model, top_n=4)
## Results
    The model outputs the input image along with the top N most similar fashion recommendations based on visual features.

## Future Work
  Add more diverse datasets to improve recommendation accuracy.
  Optimize the model for faster prediction times.
  Integrate the system into a web-based application for live recommendations.
## License
  This project is licensed under the MIT License.
