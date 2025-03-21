import streamlit as st
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import datetime
import json
import os
from pathlib import Path
import plotly.express as px
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Set page configuration
st.set_page_config(
    page_title="Smart Waste Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        padding: 2rem;
        background-color: #f8f9fa;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #3e8e41;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    .upload-box {
        border: 2px dashed #4CAF50;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        background-color: rgba(76, 175, 80, 0.05);
        transition: all 0.3s ease;
    }
    
    .upload-box:hover {
        border-color: #3e8e41;
        background-color: rgba(76, 175, 80, 0.1);
    }
    
    .info-box {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
    }
    
    .info-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    
    .gemini-box {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
        border-left: 5px solid #2196F3;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    .model-box {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
        border-left: 5px solid #9C27B0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    h1, h2, h3 {
        color: #2E7D32;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    
    /* Chat widget styles */
    .chat-widget-container {
        position: fixed !important;
        bottom: 20px;
        right: 20px;
        z-index: 9999;
    }
    
    .chat-icon {
        width: 60px;
        height: 60px;
        background-color: #4CAF50;
        border-radius: 50%;
        display: flex;
        justify-content: center;
        align-items: center;
        color: white;
        font-size: 24px;
        cursor: pointer;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .chat-icon:hover {
        transform: scale(1.1);
        background-color: #3e8e41;
    }
    
    .chat-widget {
        position: fixed !important;
        bottom: 20px;
        right: 20px;
        width: 350px;
        z-index: 9999;
        background: white;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        display: none;
    }
    
    .chat-widget.visible {
        display: block;
        animation: slideIn 0.3s forwards;
    }
    
    @keyframes slideIn {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    .chat-header {
        background-color: #4CAF50;
        color: white;
        padding: 15px;
        border-radius: 15px 15px 0 0;
        cursor: pointer;
        font-weight: bold;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .chat-close {
        cursor: pointer;
        font-size: 18px;
    }
    
    .chat-header:hover {
        background-color: #3e8e41;
    }
    
    .chat-body {
        background-color: white;
        border: 1px solid #eee;
        border-top: none;
        border-radius: 0 0 15px 15px;
        padding: 20px;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .chat-message {
        margin-bottom: 15px;
        padding: 12px;
        border-radius: 18px;
        max-width: 85%;
        word-wrap: break-word;
        animation: fadeIn 0.5s;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background-color: #e3f2fd;
        margin-left: 15%;
        color: #000;
        border-bottom-right-radius: 5px;
    }
    
    .bot-message {
        background-color: #f5f5f5;
        margin-right: 15%;
        color: #000;
        border-bottom-left-radius: 5px;
    }
    
    .chat-input {
        margin-top: 10px;
        padding: 15px;
        border-top: 1px solid #eee;
    }
    
    div[data-testid="stToolbar"] {
        display: none;
    }
    
    .footer {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Custom file uploader */
    .css-1offfwp {
        border-radius: 15px !important;
        border: 2px dashed #4CAF50 !important;
        background-color: rgba(76, 175, 80, 0.05) !important;
    }
    
    /* Input fields */
    div[data-baseweb="input"] {
        border-radius: 10px !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1e3932;
        border-right: 1px solid #e0e0e0;
    }
    
    section[data-testid="stSidebar"] > div {
        padding: 2rem 1rem;
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] div {
        color: #e0e0e0 !important;
    }
    
    /* Tabs in sidebar */
    section[data-testid="stSidebar"] [data-baseweb="tab"] {
        background-color: #2b4d44 !important;
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    
    /* Progress bars */
    div[role="progressbar"] > div {
        background-color: #4CAF50 !important;
    }
    
    /* Custom JavaScript for chat widget toggle */
    .stApp iframe[height="0"] {
        display: none;
    }
    </style>
    
    <script>
    // JavaScript to handle chat widget toggle
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(function() {
            const chatIcon = document.querySelector('.chat-icon');
            const chatWidget = document.querySelector('.chat-widget');
            const chatClose = document.querySelector('.chat-close');
            
            if (chatIcon && chatWidget) {
                chatIcon.addEventListener('click', function() {
                    chatWidget.classList.add('visible');
                    chatIcon.style.display = 'none';
                });
                
                if (chatClose) {
                    chatClose.addEventListener('click', function() {
                        chatWidget.classList.remove('visible');
                        chatIcon.style.display = 'flex';
                    });
                }
            }
        }, 1000);
    });
    </script>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Initialize session state for chat
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'chat_visible' not in st.session_state:
    st.session_state.chat_visible = False
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None

# Load the model and processor (cached for efficiency)
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained("yangy50/garbage-classification")
    model = AutoModelForImageClassification.from_pretrained("yangy50/garbage-classification")
    return processor, model

# Gemini model configuration
@st.cache_resource
def load_gemini_model():
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        return model
    except Exception as e:
        st.error(f"Error loading Gemini model: {str(e)}")
        return None

# Function to analyze image with Gemini
def analyze_with_gemini(image, model):
    try:
        prompt = """Analyze this waste image and provide the following information:
        1. Garbage Type: Identify the specific type of waste shown
        2. Classification: Categorize as recyclable, hazardous, organic, etc. with a brief description
        3. Disposal Instructions: Provide clear steps for proper disposal
        4. Confidence: Rate your confidence in this analysis (high, medium, low)
        
        Format your response as a structured list with these headings only.
        """
        
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Error analyzing image with Gemini: {str(e)}"

# Function to handle chat interactions
def process_chat_query(query, context):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""As a waste management expert, answer the following question about waste:
        Context about the waste: {context}
        User Question: {query}
        
        Provide a clear, concise, and helpful response focusing on proper waste management practices."""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"

# Enhanced class labels and disposal methods
WASTE_CATEGORIES = {
    "plastic": {
        "disposal": "Recycle in designated plastic recycling bins.",
        "impact": "Takes 450+ years to decompose",
        "subcategories": ["PET", "HDPE", "PVC", "LDPE", "PP", "PS"],
        "tips": "Rinse containers before recycling, remove caps and labels if possible."
    },
    "metal": {
        "disposal": "Recycle at metal collection points or scrap yards.",
        "impact": "Saves 95% energy compared to new metal production",
        "subcategories": ["Aluminum", "Steel", "Copper", "Tin"],
        "tips": "Crush cans to save space, separate different metal types."
    },
    "paper": {
        "disposal": "Recycle in paper recycling bins, keep dry.",
        "impact": "Saves 17 trees per ton recycled",
        "subcategories": ["Cardboard", "Newspaper", "Office Paper", "Magazines"],
        "tips": "Remove plastic windows from envelopes, flatten cardboard boxes."
    },
    "glass": {
        "disposal": "Recycle in glass-specific bins by color.",
        "impact": "100% recyclable without quality loss",
        "subcategories": ["Clear", "Green", "Brown", "Mixed"],
        "tips": "Rinse containers, remove metal caps and cork."
    },
    "organic": {
        "disposal": "Compost or use organic waste bins.",
        "impact": "Reduces methane emissions from landfills",
        "subcategories": ["Food Waste", "Garden Waste", "Coffee Grounds", "Tea Bags"],
        "tips": "Layer green and brown materials in compost."
    },
    "hazardous": {
        "disposal": "Take to specialized hazardous waste facilities.",
        "impact": "Prevents toxic substances from entering environment",
        "subcategories": ["Batteries", "Electronics", "Chemicals", "Paint"],
        "tips": "Never mix different hazardous materials."
    },
    "e-waste": {
        "disposal": "Return to electronics retailers or recycling centers.",
        "impact": "Recovers valuable rare earth metals",
        "subcategories": ["Computers", "Phones", "Appliances", "Cables"],
        "tips": "Securely erase personal data before disposal."
    },
    "textile": {
        "disposal": "Donate usable items, recycle worn textiles.",
        "impact": "Reduces landfill space and water pollution",
        "subcategories": ["Clothes", "Shoes", "Bags", "Linens"],
        "tips": "Clean items before donation, repair when possible."
    }
}

processor, model = load_model()
gemini_model = load_gemini_model()

# Sidebar
with st.sidebar:
    st.title("♻️ Smart Waste Classification")
    
    # Create tabs in sidebar
    tab1, tab2, tab3 = st.tabs(["⚙️ Settings", "📊 Stats", "ℹ️ About"])
    
    with tab1:
        st.toggle("Dark Mode", key="dark_mode", value=False)
        
        if not GOOGLE_API_KEY:
            st.error("⚠️ Gemini API key not found. Please add GOOGLE_API_KEY to your .env file.")
    
    with tab2:
        st.subheader("📊 Statistics")
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            fig = px.pie(df, names='waste_type', title='Waste Distribution')
            fig.update_layout(
                legend_title="Waste Types",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No waste classification history yet. Start analyzing images to see statistics.")
    
    with tab3:
        st.subheader("About this App")
        st.markdown("""
        This Smart Waste Classification app helps you identify and properly dispose of waste items.
        
        **Features:**
        - AI-powered waste classification
        - Detailed disposal instructions
        - Environmental impact information
        - Interactive chat assistance
        
        **How to use:**
        1. Upload an image of waste
        2. View classification results
        3. Follow disposal recommendations
        4. Use chat for additional help
        """)
        
        st.divider()
        st.caption("Version 1.0.0 | Made with ❤️ for a cleaner planet")

# Main content
st.title("🌍 Smart Waste Classification")
st.markdown("Upload an image of waste to get classification and disposal instructions from both our ML model and Gemini AI.")

# Create a container for the upload section with improved styling
upload_container = st.container()
with upload_container:
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drag and drop or click to upload", type=["jpg", "png", "jpeg"])
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner('Analyzing image...'):
        # Create two columns for the two different analyses
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="model-box">', unsafe_allow_html=True)
            st.subheader("🤖 ML Model Analysis")
            
            # Hugging Face model analysis
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get prediction
            predicted_class_idx = logits.argmax(-1).item()
            prediction_label = model.config.id2label[predicted_class_idx]
            confidence = probabilities[0][predicted_class_idx].item() * 100
            
            # Display prediction
            waste_type = prediction_label.lower()
            st.markdown(f"### Classification: {prediction_label.title()}")
            st.progress(confidence/100)
            st.markdown(f"**Confidence:** {confidence:.2f}%")
            
            # Display disposal information
            waste_info = WASTE_CATEGORIES.get(waste_type, {"disposal": "Unknown disposal method", "impact": "Unknown impact", "tips": "No specific tips available"})
            
            st.markdown("### Disposal Instructions")
            st.info(waste_info['disposal'])
            
            st.markdown("### Environmental Impact")
            st.warning(waste_info['impact'])
            
            st.markdown("### Handling Tips")
            st.success(waste_info['tips'])
            
            # Display subcategories if available
            if 'subcategories' in waste_info:
                st.markdown("### Subcategories")
                subcats = waste_info['subcategories']
                cols = st.columns(len(subcats))
                for i, subcat in enumerate(subcats):
                    with cols[i]:
                        st.markdown(f"**{subcat}**")
            
            st.markdown('</div>', unsafe_allow_html=True)

        # Gemini AI Analysis
        with col2:
            st.markdown('<div class="gemini-box">', unsafe_allow_html=True)
            st.subheader("🧠 AI Analysis")
            
            if GOOGLE_API_KEY:
                # Get Gemini analysis
                gemini_analysis = analyze_with_gemini(image, gemini_model)
                st.markdown(gemini_analysis)
                
                # Create a visualization based on waste type
                st.subheader("📊 Waste Classification Context")
                
                # Create sample data for visualization
                categories = list(WASTE_CATEGORIES.keys())
                values = [10, 15, 20, 25, 30, 35, 40, 45]  # Sample values
                
                # Highlight the predicted category
                colors = ['lightgray'] * len(categories)
                if waste_type.lower() in categories:
                    idx = categories.index(waste_type.lower())
                    colors[idx] = '#4CAF50'
                
                # Create bar chart
                fig = px.bar(
                    x=categories, 
                    y=values,
                    labels={'x': 'Waste Category', 'y': 'Volume (%)'},
                    title='Waste Distribution by Category'
                )
                
                # Customize colors
                fig.update_traces(marker_color=colors)
                
                # Add annotation for the detected waste
                if waste_type.lower() in categories:
                    fig.add_annotation(
                        x=waste_type.lower(),
                        y=values[categories.index(waste_type.lower())],
                        text="Detected",
                        showarrow=True,
                        arrowhead=1
                    )
                
                # Improve chart appearance
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Poppins, sans-serif"),
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Please configure your Gemini API key to see AI analysis")
            st.markdown('</div>', unsafe_allow_html=True)

        # Update the session state with current analysis
        st.session_state.current_analysis = f"Image shows {waste_type} with {confidence:.2f}% confidence. {waste_info['disposal'] if waste_type.lower() in WASTE_CATEGORIES else ''}"

        # Store in history
        st.session_state.history.append({
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'waste_type': waste_type,
            'confidence': confidence,
            'disposal': waste_info['disposal'] if waste_type.lower() in WASTE_CATEGORIES else 'Unknown'
        })

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("Made with ❤️ for a cleaner planet | Learn more about [waste management](https://www.epa.gov/recycle)")
st.markdown("</div>", unsafe_allow_html=True)

def render_chat_widget():
    # Create a container for the chat widget
    chat_container = st.container()
    
    with chat_container:
        # Chat icon (always visible)
        st.markdown(
            """
            <div class="chat-widget-container">
                <div class="chat-icon">💬</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Chat widget (initially hidden, shown when icon is clicked)
        st.markdown('<div class="chat-widget">', unsafe_allow_html=True)
        
        # Chat header with close button
        st.markdown(
            """
            <div class="chat-header">
                <span>Waste Management Assistant</span>
                <span class="chat-close">✕</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Chat body
        st.markdown('<div class="chat-body">', unsafe_allow_html=True)
        
        # Display chat messages
        for message in st.session_state.chat_messages:
            message_class = "user-message" if message['type'] == 'user' else "bot-message"
            st.markdown(
                f'<div class="chat-message {message_class}">{message["text"]}</div>',
                unsafe_allow_html=True
            )
        
        # Chat input section
        st.markdown('<div class="chat-input">', unsafe_allow_html=True)
        user_input = st.text_input("Type your question here:", key="chat_input", 
                                 placeholder="Ask about waste management...")
        
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("Send", key="send_button", use_container_width=True):
                if user_input.strip():
                    # Add user message
                    st.session_state.chat_messages.append({
                        'type': 'user',
                        'text': user_input
                    })
                    
                    # Get context from current analysis or use general context
                    context = st.session_state.current_analysis if st.session_state.current_analysis else "General waste management"
                    
                    # Get bot response
                    response = process_chat_query(user_input, context)
                    
                    # Add bot response
                    st.session_state.chat_messages.append({
                        'type': 'bot',
                        'text': response
                    })
                    
                    # Clear input and rerun using the current API
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Add chat widget at the end
render_chat_widget()
