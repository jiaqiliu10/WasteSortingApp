# CS5330 Final Project Waste Sorting App
This project is developed for the CS5330 course to help users classify waste items and receive recycling guidance.  The app utilizes a pre-trained machine learning model to classify waste images into different categories and provides disposal suggestions based on the classification.
## Project Team
- Jiaqi Liu
- Pingqi An
- Zhao Liu

## Overview

This project includes two main interfaces for waste classification:
- **Streamlit Web App**：A user-friendly interface for easy waste classification.
- **Hugging Face Demo**：An alternative interface hosted on Hugging Face Spaces.

## Getting Started

### Streamlit Web App
You can access the Streamlit-based version of the Waste Sorting App at the following URL: https://wastesortingapp-cs5330finalproject.streamlit.app/

### Hugging Face Demo
Alternatively, you can try the Hugging Face version of the Waste Sorting App: https://huggingface.co/spaces/jiaqiliuu/CS5330_Final_Project

### Running the Project Locally
To run the project locally, follow these steps:
1. **Clone the repository**：
   ```bash
   git clone https://github.com/jiaqiliu10/WasteSortingApp.git
2. **Install dependencies**：
   ```bash
   pip install -r requirements.txt
3. **Run the Streamlit App**：
   ```bash
   streamlit run app_streamlit.py
4. **Run the Gradio App**：
   ```bash
   python3 app.py

## File Descriptions

- **app_streamlit.py**:  
  The main file for the Streamlit web application interface. Users can upload images, select a language (English/Chinese), and receive waste classification results and recycling suggestions. This interface also allows users to download annotated images and submit feedback.

- **app.py**:  
  An alternative Gradio-based interface for waste classification. Similar to the Streamlit app, it provides classification, guidance, and feedback functionality.

- **model_inference.py**:  
  Contains the core inference logic for waste classification. This file includes functions to preprocess images, perform model inference, and apply a confidence threshold to classify low-confidence results as "trash."

- **requirements.txt**:  
  Lists all necessary Python packages to run the app. Install these dependencies to ensure the app functions correctly.

- **user_feedback.txt**:  
  Stores user feedback locally, allowing developers to review and improve classification accuracy.

- **test_picture**:  
  A folder containing sample images that can be used to test the app's waste classification functionality. Users can upload these images to evaluate how the model classifies different types of waste and provides recycling guidance.

## Usage Instructions

- **Upload Image**: Upload a clear image of the waste item for classification.
- **Select Language**: Choose between English and Chinese for the classification description.
- **Classify Waste**: Receive classification results and recycling guidance.
- **Download Annotated Image**: Save an annotated image showing the classification result.
- **Submit Feedback**: Provide feedback on the classification result to help improve the model's accuracy.

## Model Information

The model used in this project is **ViT TrashNet Enhanced**, trained to recognize various types of waste. The model is hosted on Hugging Face and can classify items into the following categories:

- Biodegradable
- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash (for items that do not fit other categories or have low confidence)

## Contact Information

For any questions or suggestions, please contact the project team members:

- **Jiaqi Liu**: liu.jiaqi10@northeastern.edu
- **Pingqi An**: an.p@northeastern.edu
- **Zhao Liu**: liu.zhao3@northeastern.edu
