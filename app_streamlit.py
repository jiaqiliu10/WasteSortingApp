# CS 5330 Final Project
# Automated Waste Classification and Recycling Guidance Assistant
# Jiaqi Liu/ Pingqi An/ Zhao Liu
# Nov 15 2024
# This file implements a Streamlit-based user interface for waste classification and recycling guidance.
# Users can upload images of waste items and select a preferred language (English/Chinese).
# It allows users to download annotated images with classification details and guidance.
# Additionally, users can submit feedback on the classification results, which can be saved locally or emailed.

import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image, ImageDraw, ImageFont
import torch
import io
import datetime
import urllib.parse

# Load model and image processor
image_processor = AutoImageProcessor.from_pretrained(
    "edwinpalegre/ee8225-group4-vit-trashnet-enhanced"
)
model = AutoModelForImageClassification.from_pretrained(
    "edwinpalegre/ee8225-group4-vit-trashnet-enhanced"
)
# Classification labels and descriptions
trash_classes = [
    "biodegradable", "cardboard", "glass", "metal", "paper", "plastic", "trash"
]
class_descriptions = {
    "English": {
        "biodegradable": (
            "Biodegradable waste such as food scraps and plant residues, "
            "can decompose naturally in soil."
        ),
        "cardboard": (
            "Cardboard waste like shipping boxes, can be recycled into new cardboard."
        ),
        "glass": (
            "Glass waste can be placed in glass recycling bins for reuse."
        ),
        "metal": (
            "Metal waste like cans and tins, can be recycled to create new metal products."
        ),
        "paper": (
            "Paper waste, such as newspapers and office paper, "
            "can be recycled into new paper products."
        ),
        "plastic": (
            "Plastic waste should be disposed of in the plastic recycling bin, "
            "if recyclable."
        ),
        "trash": (
            "General waste that doesn't fit into other categories; "
            "should be disposed of in regular waste bins."
        )
    },
    "Chinese": {
        "biodegradable": (
            "可生物降解垃圾，如厨余和植物残渣，可以在土壤中自然分解。"
        ),
        "cardboard": (
            "纸板废物，如快递盒，可以回收处理为新纸板。"
        ),
        "glass": (
            "玻璃废物，可以放入玻璃回收箱，循环利用。"
        ),
        "metal": (
            "金属废物如罐头，可以回收用来制造新金属产品。"
        ),
        "paper": (
            "纸张废物，如报纸和办公用纸，可以回收成新纸制品。"
        ),
        "plastic": (
            "塑料垃圾应放入塑料回收箱（如可回收）。"
        ),
        "trash": (
            "不属于其他类别的一般废物，应放入普通垃圾箱。"
        )
    }
}

# Color mapping for categories
category_colors = {
    "glass": "blue",
    "metal": "blue",
    "paper": "blue",
    "plastic": "blue",
    "biodegradable": "green",
    "trash": "black",
    "cardboard": "blue"
}

# Preprocess the uploaded image for the model
def preprocess_image(image):
    if image.mode != "RGB":  # Ensure the image is in RGB format
        image = image.convert("RGB")
    # Convert image to model input
    inputs = image_processor(images=image, return_tensors="pt")  
    return inputs

# Classify the image and handle low-confidence predictions
def classify_image_with_trash_threshold(image, threshold=0.7):
    inputs = preprocess_image(image)  # Preprocess the image
    with torch.no_grad():  # Disable gradient calculations for inference
        outputs = model(**inputs)  # Get model predictions
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[0]  # Calculate probabilities
        predicted_class = probabilities.argmax().item()  # Get the class with the highest probability
        confidence = probabilities[predicted_class].item()  # Confidence of the prediction

    # Assign "trash" if confidence is below the threshold
    if confidence < threshold:
        class_name = "trash"
        confidence = threshold
    else:
        class_name = trash_classes[predicted_class]

    return class_name, confidence

# Annotate the image with classification results
def annotate_image(image, class_name, confidence, guidance):
    annotated_image = image.copy()  # Create a copy of the image
    draw = ImageDraw.Draw(annotated_image)  # Initialize drawing context
    text = (
        f"Classification: {class_name}\n"
        f"Confidence: {confidence:.2f}\n"
        f"{guidance}"
    )  # Annotation text

    try:
        # Load a font for the annotation text
        font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 30)
    except IOError:
        # Fallback to default font if Arial is unavailable
        font = ImageFont.load_default()

    # Break text into lines to fit within the image width
    max_width = annotated_image.width - 20
    lines = []
    line = ""
    for word in text.split():
        test_line = f"{line} {word}".strip()
        text_width = draw.textbbox((0, 0), test_line, font=font)[2]
        if text_width <= max_width:
            line = test_line
        else:
            lines.append(line)
            line = word
    lines.append(line)

    # Draw each line of text on the image
    y_text = 10
    for line in lines:
        draw.text((10, y_text), line, fill="red", font=font)  # Add red text to the image
        y_text += draw.textbbox((0, 0), line, font=font)[3] + 5

    # Save the annotated image in memory as a PNG
    img_byte_arr = io.BytesIO()
    annotated_image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    return img_byte_arr

# CSS for styling the Streamlit app
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 30px;
        font-weight: bold;
        color: #4CAF50;
        margin-bottom: 20px;
    }
    .instruction {
        font-size: 18px;
        text-align: left;
        color: #555;
        margin: 20px 0;
    }
    .result-box {
        border: 1px solid #ddd;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        background-color: #f9f9f9;
    }
    .feedback-box {
        margin-top: 30px;
        padding: 20px;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #f1f1f1;
    }
    .color-legend {
        font-size: 16px;
        margin-top: 20px;
    }
    .blue { color: blue; font-weight: bold; }
    .green { color: green; font-weight: bold; }
    .black { color: black; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True
)

# Add a title and instructions to the app
st.markdown(
    "<div class='title'>Automated Waste Sorting Assistant</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div class='instruction'>Upload an image to receive waste sorting suggestions.</div>",
    unsafe_allow_html=True
)
# Add detailed instructions and a color legend
st.markdown("""
### Usage Instructions:
1. **Upload an image of the waste item.**
2. **Select your preferred language (English/Chinese).**
3. **Click "Classify Waste" to get the classification result and disposal guidance.**

**Note:**  
Some special items may be misclassified as "trash". If you find an incorrect classification, please provide feedback to help us improve.

### Color Legend:
<div class="color-legend">
    <p><span class="blue">Blue</span> - Recyclable materials (e.g., glass, metal, paper, plastic).</p>
    <p><span class="green">Green</span> - Biodegradable waste (e.g., food scraps).</p>
    <p><span class="black">Black</span> - General waste (e.g., non-recyclable trash).</p>
</div>
""", unsafe_allow_html=True)

# Allow the user to upload an image and select a language
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_image = st.file_uploader(
        "Upload an image of waste...", type=["jpg", "jpeg", "png"]
    )
    
with col2:
    language = st.radio("Select language", ("English", "Chinese"))

# Display the classification result
if uploaded_image:
    image = Image.open(uploaded_image)  # Open the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)  # Show the uploaded image

    # Perform classification
    class_name, confidence = classify_image_with_trash_threshold(image)
    description = class_descriptions[language].get(class_name, "No description available.")  # Get description
    color = category_colors.get(class_name, "black")  # Get category color

    # Display classification details
    st.markdown(f"<div class='result-box'><h3 style='color:{color}'>{class_name}</h3>"
                f"<p>Confidence: {confidence:.2f}</p>"
                f"<p>Description: {description}</p></div>", unsafe_allow_html=True)

    # Annotate the image and allow download
    guidance = (
        f"Suggestion: Place {class_name} in the appropriate recycling bin."
        if language == "English"
        else f"建议：将 {class_name} 放入对应的回收箱。"
    )
    img_byte_arr = annotate_image(image, class_name, confidence, guidance)

    st.download_button(
        label="Download Annotated Image",
        data=img_byte_arr,
        file_name="classification_result.png",
        mime="image/png"
    )

# Feedback section
st.markdown("<div class='feedback-box'><h4>Feedback</h4></div>", unsafe_allow_html=True)
feedback = st.text_input("Provide feedback if classification was incorrect:")

# Handle feedback submission
if st.button("Submit Feedback"):
    feedback_entry = (
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
        f"Feedback: {feedback}\n"
    )
    # Save feedback locally
    with open("user_feedback.txt", "a") as file:
        file.write(feedback_entry)
    
    # Create a mailto link for feedback
    email = "wastesortingapp@gmail.com"
    subject = "Feedback"
    body = urllib.parse.quote(feedback_entry)  # URL-encode feedback
    mailto_link = f"mailto:{email}?subject={subject}&body={body}"
    
    # Show feedback success message and email link
    st.markdown(
        f'<a href="{mailto_link}" target="_blank">'
        f'Click here to send feedback via email</a>',
        unsafe_allow_html=True
    )
    st.success(
        "Thank you for your feedback! Your feedback has been saved and can be sent via email."
    )
