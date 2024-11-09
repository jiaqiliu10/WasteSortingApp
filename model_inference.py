# model_inference.py
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import torch

# Load the feature extractor and model
feature_extractor = AutoFeatureExtractor.from_pretrained("edwinpalegre/ee8225-group4-vit-trashnet-enhanced")
model = AutoModelForImageClassification.from_pretrained("edwinpalegre/ee8225-group4-vit-trashnet-enhanced")

# Waste classification labels and descriptions
trash_classes = ["biodegradable", "cardboard", "glass", "metal", "paper", "plastic", "trash"]
class_descriptions = {
    "English": {
        "biodegradable": "Biodegradable waste such as food scraps and plant residues, can decompose naturally in soil.",
        "cardboard": "Cardboard waste like shipping boxes, can be recycled into new cardboard.",
        "glass": "Glass waste can be placed in glass recycling bins for reuse.",
        "metal": "Metal waste like cans and tins, can be recycled to create new metal products.",
        "paper": "Paper waste, such as newspapers and office paper, can be recycled into new paper products.",
        "plastic": "Plastic waste should be disposed of in the plastic recycling bin, if recyclable.",
        "trash": "General waste that doesn't fit into other categories; should be disposed of in regular waste bins."
    },
    "Chinese": {
        "biodegradable": "可生物降解垃圾，如厨余和植物残渣，可以在土壤中自然分解。",
        "cardboard": "纸板废物，如快递盒，可以回收处理为新纸板。",
        "glass": "玻璃废物，可以放入玻璃回收箱，循环利用。",
        "metal": "金属废物如罐头，可以回收用来制造新金属产品。",
        "paper": "纸张废物，如报纸和办公用纸，可以回收成新纸制品。",
        "plastic": "塑料垃圾应放入塑料回收箱（如可回收）。",
        "trash": "不属于其他类别的一般废物，应放入普通垃圾箱。"
    }
}

# Image preprocessing function
def preprocess_image(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs


# Classification function with confidence threshold for trash
def classify_image_with_trash_threshold(image, threshold=0.7):
    inputs = preprocess_image(image)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_class = probabilities.argmax().item()
        confidence = probabilities[predicted_class].item()
    
    # Apply threshold to classify low-confidence predictions as "trash"
    if confidence < threshold:
        class_name = "trash"
        confidence = threshold  # Set a default confidence for trash classification
    else:
        class_name = trash_classes[predicted_class]
    
    return class_name, confidence
