import gradio as gr
from model_inference import classify_image_with_trash_threshold, class_descriptions
from PIL import Image, ImageDraw, ImageFont
import datetime

# Define color mapping for categories
category_colors = {
    "glass": "blue",
    "metal": "blue",
    "paper": "blue",
    "plastic": "blue",
    "biodegradable": "green",
    "trash": "black",
    "cardboard": "blue"  # Assuming cardboard is recyclable in Vancouver
}

# Classification and suggestion output function with color-coded results
def waste_sorting(image, language):
    class_name, confidence = classify_image_with_trash_threshold(image)
    guidance = f"Suggestion: Place {class_name} in the appropriate recycling bin. Confidence: {confidence:.2f}" if language == "English" else f"建议：将 {class_name} 放入对应的回收箱。置信度: {confidence:.2f}"
    description = class_descriptions[language].get(class_name, "No description available." if language == "English" else "没有可用的描述。")
    
    # Get the color for the class_name category
    color = category_colors.get(class_name, "black")  # Default to black if category not found
    class_name_html = f"<div style='color: {color}; font-weight: bold;'>{class_name}</div>"
    return class_name_html, guidance, confidence, description

def download_result(image, class_name, confidence, guidance):
    # Remove any HTML tags if present in the class_name
    plain_class_name = class_name.split('>')[-2].split('<')[0].strip()  # Extract the text without HTML tags

    # Convert numpy array to PIL image
    annotated_image = Image.fromarray(image)
    draw = ImageDraw.Draw(annotated_image)
    
    # Define text to display
    text = f"Classification: {plain_class_name}\nConfidence: {confidence:.2f}\n{guidance}"

    # Set font size and wrapping
    try:
        font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 20)  # macOS example
    except IOError:
        font = ImageFont.load_default()  # Fallback if the specified font is not found

    # Wrap text to fit within image width
    max_width = annotated_image.width - 20  # Set some padding
    lines = []
    line = ""
    for word in text.split():
        test_line = f"{line} {word}".strip()
        # Use textbbox instead of textsize to measure text width
        if draw.textbbox((0, 0), test_line, font=font)[2] <= max_width:
            line = test_line
        else:
            lines.append(line)
            line = word
    lines.append(line)  # Add the last line

    # Draw wrapped text on the image
    y_text = 10  # Start position for the text
    for line in lines:
        draw.text((10, y_text), line, font=font, fill="white")
        y_text += draw.textbbox((0, 0), line, font=font)[3] + 5  # Add some padding between lines

    # Save annotated image
    output_path = "classification_result_with_text.png"
    annotated_image.save(output_path)
    return output_path



# Function to handle user feedback
def submit_feedback(image, feedback_text):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    feedback_entry = f"{timestamp} - Feedback: {feedback_text}\n"
    
    # Append the feedback to a text file
    with open("user_feedback.txt", "a") as file:
        file.write(feedback_entry)
    
    return "Thank you for your feedback!"

# Create Gradio interface with color coding, instructions, and feedback mechanism
with gr.Blocks() as iface:
    gr.Markdown("<h1 style='text-align: center;'>Automated Waste Sorting Assistant</h1>")
    gr.Markdown("""
    <p style='text-align: center;'>Upload an image to receive waste sorting suggestions.</p>
    <h3>Usage Instructions:</h3>
    <ul>
        <li>Upload an image of the waste item.</li>
        <li>Select your preferred language (English/Chinese).</li>
        <li>Click "Classify Waste" to get the classification result and disposal guidance.</li>
    </ul>
    <h3>Note:</h3>
    <p>Some special items may be misclassified as "trash". If you find an incorrect classification, please provide feedback to help us improve.</p>
    <h3>Color Legend:</h3>
    <ul>
        <li><span style='color: blue;'>Blue</span> - Recyclable materials (e.g., glass, metal, paper, plastic).</li>
        <li><span style='color: green;'>Green</span> - Biodegradable waste (e.g., food scraps).</li>
        <li><span style='color: black;'>Black</span> - General waste (e.g., non-recyclable trash).</li>
    </ul>
    """)

    with gr.Row():
        # Left column for image input and language selection
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Image")
            language_radio = gr.Radio(["English", "Chinese"], label="Language", value="English")
            waste_sort_button = gr.Button("Classify Waste")
        
        # Right column for displaying results
        with gr.Column(scale=1):
            class_name_output = gr.HTML(label="Class Name")  # Using HTML to show colored text
            guidance_output = gr.Textbox(label="Suggestion", interactive=False)
            confidence_output = gr.Slider(minimum=0, maximum=1, step=0.01, label="Confidence", interactive=False)
            description_output = gr.Textbox(label="Waste Description", interactive=False)
            
            # Download button for annotated image
            download_button = gr.Button("Download Result")

    # Define feedback submission components
    feedback_input = gr.Textbox(label="Feedback", placeholder="Enter your feedback if classification is incorrect")
    feedback_button = gr.Button("Submit Feedback")
    feedback_output = gr.Textbox(label="Feedback Status", interactive=False)

    # Define actions for buttons
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

# Launch the Gradio interface
iface.launch(share=True)
