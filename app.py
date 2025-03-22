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
import random

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
        width: 380px;
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
        background: linear-gradient(135deg, #43a047 0%, #1b5e20 100%);
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
        background: linear-gradient(135deg, #388e3c 0%, #1a5e20 100%);
    }
    
    .chat-body {
        background-color: #f9f9f9;
        padding: 15px;
        height: 350px;
        overflow-y: auto;
        border-bottom: 1px solid #eee;
    }
    
    .chat-message {
        margin-bottom: 15px;
        padding: 12px 15px;
        border-radius: 12px;
        max-width: 85%;
        word-wrap: break-word;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        line-height: 1.5;
        font-size: 0.95em;
    }
    
    .user-message {
        background-color: #E8F5E9;
        margin-left: auto;
        border-bottom-right-radius: 4px;
        color: #1e3932;
        font-weight: 500;
    }
    
    .bot-message {
        background-color: white;
        margin-right: auto;
        border-bottom-left-radius: 4px;
        color: #333;
        border-left: 3px solid #4CAF50;
    }
    
    .chat-input {
        padding: 15px;
        background-color: white;
        border-radius: 0 0 15px 15px;
    }
    
    .chat-input input {
        border-radius: 20px;
        border: 1px solid #ddd;
        padding: 12px 15px;
        width: 100%;
        font-size: 0.95em;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .chat-input button {
        border-radius: 20px;
        background: linear-gradient(135deg, #43a047 0%, #2e7d32 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.85em;
    }
    
    .chat-input button:hover {
        background: linear-gradient(135deg, #388e3c 0%, #1b5e20 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }
    
    footer {
        visibility: hidden;
    }
    
    header {
        visibility: hidden;
    }
    
    div[data-testid="stToolbar"] {
        display: none;
    }
    
    .footer {
        text-align: center;
        padding: 20px;
        color: #666;
        font-size: 0.8em;
    }
    
    /* Team section styling */
    .team-section {
        background: linear-gradient(135deg, #43a047 0%, #1b5e20 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    
    .team-title {
        color: white !important;
        text-align: center;
        margin-bottom: 15px;
        font-weight: 700 !important;
    }
    
    .team-members {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 15px;
    }
    
    .team-member {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
        flex: 1 1 calc(33.333% - 15px);
        min-width: 200px;
    }
    
    .team-member:hover {
        background-color: rgba(255, 255, 255, 0.2);
        transform: translateY(-5px);
    }
    
    .team-member-name {
        font-weight: 600;
        font-size: 1.1em;
        margin-bottom: 5px;
    }
    
    .team-member-id {
        opacity: 0.8;
    }
    
    /* Waste theme elements */
    .recycle-icon {
        font-size: 2em;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .waste-category-icon {
        font-size: 1.5em;
        margin-right: 10px;
    }
    
    .app-header {
        background: linear-gradient(135deg, #43a047 0%, #1b5e20 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    
    .app-title {
        color: white !important;
        font-size: 2.5em !important;
        font-weight: 700 !important;
        margin-bottom: 10px !important;
    }
    
    .app-subtitle {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1.2em !important;
        font-weight: 400 !important;
    }
    
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = ""
if 'waste_quotes' not in st.session_state:
    st.session_state.waste_quotes = [
        "Waste isn't waste until we waste it.",
        "There is no such thing as 'away'. When we throw anything away, it must go somewhere.",
        "The greatest threat to our planet is the belief that someone else will save it.",
        "Recycling turns things into other things, which is like magic!",
        "Reduce, Reuse, Recycle - the three Rs of sustainable waste management.",
        "The Earth is what we all have in common. Let's keep it clean.",
        "Every time you recycle, the Earth smiles a little.",
        "Small acts, when multiplied by millions of people, can transform the world.",
        "Waste management is not just about managing waste, it's about managing resources.",
        "A clean environment is a happy environment."
    ]

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
        if not GOOGLE_API_KEY:
            return "Sorry, I can't process your query without a valid API key. Please add your Gemini API key to the .env file."
        
        # Prepare prompt with context
        prompt = f"Context: {context}\n\nUser Query: {query}\n\nProvide a helpful response about waste management and classification. Be concise and informative."
        
        # Generate response
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

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

# App Header with Team Details
st.markdown('<div class="app-header">', unsafe_allow_html=True)
st.markdown('<h1 class="app-title">🌍 Smart Waste Classification</h1>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Upload an image of waste to get classification and disposal instructions</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Team Section
st.markdown('<div class="team-section">', unsafe_allow_html=True)
st.markdown('<h2 class="team-title">👥 Our Team</h2>', unsafe_allow_html=True)
st.markdown('<div class="team-members">', unsafe_allow_html=True)

team_members = [
    {"name": "N. Manu", "id": "99220041537"},
    {"name": "R Venkata Vignesh", "id": "99220041334"},
    {"name": "V Bhanu Prakash", "id": "99220041025"},
    {"name": "P Sreenivasulu", "id": "9922005060"},
    {"name": "S Sai Kousick", "id": "99220041007"},
    {"name": "I Inbathamizhan", "id": "99220003016"}
]

for member in team_members:
    st.markdown(f'''
    <div class="team-member">
        <div class="recycle-icon">♻️</div>
        <div class="team-member-name">{member["name"]}</div>
        <div class="team-member-id">{member["id"]}</div>
    </div>
    ''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

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

# Upload section with improved styling
st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.markdown('<div class="waste-category-icon">📸</div> <span style="font-size: 1.3em; font-weight: 600;">Upload Waste Image</span>', unsafe_allow_html=True)
st.markdown('<p>Take a photo of waste items and our AI will classify them and provide disposal instructions</p>', unsafe_allow_html=True)

upload_container = st.container()
with upload_container:
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drag and drop or click to upload", type=["jpg", "png", "jpeg"])
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner('Analyzing image...'):
        # Display a random quote about waste handling
        quote_placeholder = st.empty()
        quote_placeholder.info(f"**Waste Wisdom:** {random.choice(st.session_state.waste_quotes)}")
        
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
st.markdown("Learn more about [waste management](https://www.epa.gov/recycle)")
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
                <span>♻️ Waste Management Assistant</span>
                <span class="chat-close">✕</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Chat body
        st.markdown('<div class="chat-body" id="chat-body">', unsafe_allow_html=True)
        
        # Display chat messages
        if st.session_state.chat_messages:
            for message in st.session_state.chat_messages:
                message_class = "user-message" if message['type'] == 'user' else "bot-message"
                st.markdown(
                    f'<div class="chat-message {message_class}">{message["text"]}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div class="chat-message bot-message">Hello! I\'m your waste management assistant. Ask me anything about waste classification, recycling, or disposal methods.</div>',
                unsafe_allow_html=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input section
        st.markdown('<div class="chat-input">', unsafe_allow_html=True)
        user_input = st.text_input("", key="chat_input", 
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
        
        # Add JavaScript for chat widget functionality
        st.markdown(
            """
            <script>
                // Function to toggle chat widget visibility
                function toggleChatWidget() {
                    const chatIcon = document.querySelector('.chat-icon');
                    const chatWidget = document.querySelector('.chat-widget');
                    
                    if (chatIcon && chatWidget) {
                        chatIcon.addEventListener('click', function() {
                            chatWidget.style.display = 'block';
                            chatIcon.style.display = 'none';
                            
                            // Scroll chat to bottom
                            const chatBody = document.getElementById('chat-body');
                            if (chatBody) {
                                chatBody.scrollTop = chatBody.scrollHeight;
                            }
                        });
                        
                        const chatClose = document.querySelector('.chat-close');
                        if (chatClose) {
                            chatClose.addEventListener('click', function() {
                                chatWidget.style.display = 'none';
                                chatIcon.style.display = 'flex';
                            });
                        }
                    }
                    
                    // Auto-scroll chat to bottom
                    const chatBody = document.getElementById('chat-body');
                    if (chatBody) {
                        chatBody.scrollTop = chatBody.scrollHeight;
                    }
                }
                
                // Run when DOM is fully loaded
                if (document.readyState === 'loading') {
                    document.addEventListener('DOMContentLoaded', function() {
                        setTimeout(toggleChatWidget, 1000);
                    });
                } else {
                    setTimeout(toggleChatWidget, 1000);
                }
            </script>
            """,
            unsafe_allow_html=True
        )

# Add chat widget at the end
render_chat_widget()
