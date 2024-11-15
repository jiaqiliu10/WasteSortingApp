# CS 5330 Final Project
# Automated Waste Classification and Recycling Guidance Assistant
# Jiaqi Liu/ Pingqi An/ Zhao Liu
# Nov 15 2024
# This file implements a Gradio-based user interface for waste classification and recycling guidance.
# Users can upload waste images, select a language, and receive classification results and disposal suggestions.
# The file also includes a user feedback submission feature and the ability to download 
# annotated images with classification information.

import gradio as gr
from model_inference import classify_image_with_trash_threshold, class_descriptions
from PIL import Image, ImageDraw, ImageFont
import datetime
import urllib.parse

# Define color mapping for different waste categories
category_colors = {
    "glass": "blue",
    "metal": "blue",
    "paper": "blue",
    "plastic": "blue",
    "biodegradable": "green",
    "trash": "black",
    "cardboard": "blue"
}

# Function to classify waste and generate suggestions
def waste_sorting(image, language):
    # Classify the uploaded image
    class_name, confidence = classify_image_with_trash_threshold(image)
    # Generate suggestions based on classification and language
    if language == "English":
        guidance = (f"Suggestion: Place {class_name} in the appropriate recycling bin. "
                    f"Confidence: {confidence:.2f}")
        description = class_descriptions[language].get(
            class_name, "No description available."
        )
    else:
        guidance = (f"建议：将 {class_name} 放入对应的回收箱。"
                    f"置信度: {confidence:.2f}")
        description = class_descriptions[language].get(
            class_name, "没有可用的描述。"
        )
    # Set the color for the category and create styled HTML for display
    color = category_colors.get(class_name, "black")
    class_name_html = f"<div style='color: {color}; font-weight: bold;'>{class_name}</div>"
    return class_name_html, guidance, confidence, description

# Function to create and save an annotated image
def download_result(image, class_name, confidence, guidance):
    # Extract plain class name from styled HTML
    plain_class_name = class_name.split('>')[-2].split('<')[0].strip()
    # Convert the input image to PIL format
    annotated_image = Image.fromarray(image)
    draw = ImageDraw.Draw(annotated_image)

    # Prepare text for annotation
    text = (f"Classification: {plain_class_name}\nConfidence: {confidence:.2f}\n"
            f"{guidance}")
    try:
        # Try to use Arial font; fallback to default if unavailable
        font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()

    # Break text into lines to fit within image width
    max_width = annotated_image.width - 20
    lines = []
    line = ""
    for word in text.split():
        test_line = f"{line} {word}".strip()
        if draw.textbbox((0, 0), test_line, font=font)[2] <= max_width:
            line = test_line
        else:
            lines.append(line)
            line = word
    lines.append(line)

    # Draw the lines of text on the image
    y_text = 10
    for line in lines:
        draw.text((20, y_text), line, font=font, fill="red")
        y_text += draw.textbbox((0, 0), line, font=font)[3] + 5

    # Save the annotated image
    output_path = "classification_result_with_text.png"
    annotated_image.save(output_path)
    return output_path

# Function to handle feedback from users
def submit_feedback(feedback_text):
    # Record the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    feedback_entry = f"{timestamp} - Feedback: {feedback_text}\n"

    # Save feedback to a text file
    with open("user_feedback.txt", "a") as file:
        file.write(feedback_entry)

    # Create a mailto link for sending feedback via email
    email = "wastesortingapp@gmail.com"
    subject = "User Feedback"
    body = urllib.parse.quote(feedback_entry)
    mailto_link = f"mailto:{email}?subject={subject}&body={body}"

    # Return a clickable HTML link for email feedback
    mailto_html = (f"<a href='{mailto_link}' target='_blank'>"
                   f"Click here to send feedback via email</a>")
    return f"Thank you for your feedback! {mailto_html}"

# Gradio interface setup
with gr.Blocks() as iface:
    # Header and instructions
    gr.HTML("""
    <style>
        h1 { text-align: center; }
    </style>
    <h1>Automated Waste Sorting Assistant</h1>
    <p style='text-align: center;'>Upload an image to receive waste sorting suggestions.</p>
    <h3>Usage Instructions:</h3>
    <ul>
        <li>Upload an image of the waste item.</li>
        <li>Select your preferred language (English/Chinese).</li>
        <li>Click "Classify Waste" to get the classification result and disposal guidance.</li>
    </ul>
    <h3>Note:</h3>
    <p>Some special items may be misclassified as "trash". If you find an incorrect classification, 
    please provide feedback to help us improve.</p>
    <h3>Color Legend:</h3>
    <ul>
        <li>🔵 **Blue** - Recyclable materials (e.g., glass, metal, paper, plastic).</li>
        <li>🟢 **Green** - Biodegradable waste (e.g., food scraps).</li>
        <li>⚫ **Black** - General waste (e.g., non-recyclable trash).</li>
    </ul>
    """)

    # Input section for image upload and language selection
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Image")
            language_radio = gr.Radio(["English", "Chinese"], label="Language", value="English")
            waste_sort_button = gr.Button("Classify Waste")

        # Output section for results
        with gr.Column(scale=1):
            class_name_output = gr.HTML(label="Class Name")
            guidance_output = gr.Textbox(label="Suggestion", interactive=False)
            confidence_output = gr.Slider(minimum=0, maximum=1, step=0.01, 
                                           label="Confidence", interactive=False)
            description_output = gr.Textbox(label="Waste Description", interactive=False)
            download_button = gr.Button("Download Result")

    # Feedback section
    feedback_input = gr.Textbox(label="Feedback", 
                                placeholder="Enter your feedback if classification is incorrect")
    feedback_button = gr.Button("Submit Feedback")
    feedback_output = gr.HTML(label="Feedback Status")

    # Link buttons to respective functions
    waste_sort_button.click(
        fn=waste_sorting,
        inputs=[image_input, language_radio],
        outputs=[class_name_output, guidance_output, confidence_output, description_output]
    )

    download_button.click(
        fn=download_result,
        inputs=[image_input, class_name_output, confidence_output, guidance_output],
        outputs=gr.File(label="Download Annotated Image")
    )

    feedback_button.click(
        fn=submit_feedback,
        inputs=feedback_input,
        outputs=feedback_output
    )

# Launch the Gradio app
iface.launch(share=True)
