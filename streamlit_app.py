import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from huggingface_hub import InferenceClient
import os
import requests
from PIL import Image
import io
from streamlit.components.v1 import html

# Load environment variables
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize the InferenceClient
client = InferenceClient(
    provider="hf-inference",
    api_key=HUGGINGFACEHUB_API_TOKEN,)

st.set_page_config(page_title="Image to Story", page_icon="📖", layout="wide")

# Inject custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #e1c8f7;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .stFileUploader>div>div>div>button {
        background-color: #2196F3;
        color: white;
    }
    .result-box {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
     .text-box {
        background-color: white;
        border-radius: 10px;
        padding: 5px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def img2text(image_data):
    """Convert image to text description"""
    output = client.image_to_text(image_data, model="Salesforce/blip-image-captioning-base")
    return output.generated_text

def generate_story(scenario):
    """Generate a short story based on the scenario"""
    template = """
    You are a story teller;
    You can generate a philosophical story based on a simple narrative, the story should not be more than 60 words;
    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)
    chain = prompt | llm
    story = chain.invoke({"scenario": scenario})
    return story.content

def text2speech(message):
    """Convert text to speech and return the audio file"""
    try:
        API_URL = "https://router.huggingface.co/hf-inference/models/espnet/kan-bayashi_ljspeech_vits"
        headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
        response = requests.post(API_URL, headers=headers, json={"inputs": message}, timeout=15)
        response.raise_for_status()  # Raises HTTPError for bad responses
        return response.content
    except Exception as e:
        st.error(f"Audio generation failed: {str(e)}")
        return None
    
# Streamlit UI

st.title("✨📖 Image to Story Magic")
st.markdown("""
<div style='background-color: #e3f2fd; padding: 2px 20px; border-radius: 10px; margin-bottom: 15px;'>
    <h3 style='color: #0d47a1;'>Upload an image and watch it transform into a captivating story with audio!</h3>
</div>
""", unsafe_allow_html=True)
       
# Initialize session state for results
if "scenario" not in st.session_state:
    st.session_state.scenario = None
if "story" not in st.session_state:
    st.session_state.story = None
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None

# File uploader
uploaded_file = st.file_uploader("Upload your file", type=["jpg", "jpeg", "png"], key="file_uploader", label_visibility="collapsed" )

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])
    if 'last_file' not in st.session_state or st.session_state.last_file != uploaded_file.name:
        st.session_state.scenario = None
        st.session_state.story = None
        st.session_state.audio_data = None
        st.session_state.last_file = uploaded_file.name

    with col1:
        st.image(uploaded_file, caption="Your Image", use_container_width=True)
        img_bytes = io.BytesIO()
        image = Image.open(uploaded_file)
        image.save(img_bytes, format="JPEG")
        img_bytes = img_bytes.getvalue()
    
    with col2:
        # Image Description Section
        if st.session_state.scenario is None:  # Only compute if not already done
            with st.spinner("🔍 Analyzing your image..."):
                scenario = img2text(img_bytes)
                st.session_state.scenario = scenario
        
        st.markdown(f"""
        <div class='result-box'>
            <h3 style='color: #2c3e50; margin-top: 0;'>📝 Image Description</h3>
            <p>{st.session_state.scenario}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Story Section
        if st.session_state.story is None:  # Only compute if not already done
            with st.spinner("✍️ Writing a magical story..."):
                story = generate_story(st.session_state.scenario)
                st.session_state.story = story
        
        st.markdown(f"""
        <div class='result-box'>
            <h3 style='color: #2c3e50; margin-top: 0;'>📖 Generated Story</h3>
            <p>{st.session_state.story}</p>
        </div>
        """, unsafe_allow_html=True)

        # Audio Section
        if st.session_state.audio_data is None:  # Only compute if not already done
            with st.spinner("🎙️ Creating audio version..."):
                audio_data = text2speech(st.session_state.story)
                st.session_state.audio_data = audio_data

        st.markdown("""
        <div class='result-box'>
            <h3 style='color: #2c3e50; margin-top: 0;'>🔊 Audio Story</h3>
        </div>
        """, unsafe_allow_html=True)

        # Create a container for the audio player & Display audio in the container
        audio_container = st.empty()
        audio_container.audio(st.session_state.audio_data, format="audio/wav")

        # Replay button logic
        if st.button("Replay Audio"):
            st.markdown(f"""
            <script>
                var audio = document.querySelector("audio");
                if (audio) {{
                    audio.currentTime = 0;
                    audio.play();
                }}
            </script>
            """, unsafe_allow_html=True)