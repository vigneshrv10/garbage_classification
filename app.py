import streamlit as st
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

# Load the model and processor (cached for efficiency)
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained("yangy50/garbage-classification")
    model = AutoModelForImageClassification.from_pretrained("yangy50/garbage-classification")
    return processor, model

processor, model = load_model()

# Streamlit UI
st.title("Garbage Classification")

uploaded_file = st.file_uploader("Upload a garbage image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # Ensure it's in RGB format
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")  

    # Perform classification
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()

    # **Get the actual class labels from the Hugging Face model card**
    class_labels = model.config.id2label  # Dictionary mapping index → label

    if predicted_class in class_labels:
        waste_type = class_labels[predicted_class]
    else:
        waste_type = "Unknown"

    # Define disposal methods (you can adjust based on actual class names)
    disposal_methods = {
        "plastic": "Recycle plastic bottles, bags, and containers.",
        "metal": "Recycle cans and scrap metals at collection centers.",
        "paper": "Recycle paper, cardboard, and books in a dry bin.",
        "glass": "Reuse or recycle glass bottles and jars.",
        "organic": "Compost organic waste like food scraps and leaves.",
        "hazardous": "Dispose of batteries, chemicals, and e-waste at hazardous waste facilities."
    }

    # Get disposal method (default if class not found)
    method = disposal_methods.get(waste_type.lower(), "No disposal method available.")

    # Display output
    st.success(f"**Waste Type:** {waste_type}")
    st.warning(f"**Disposal Method:** {method}")
