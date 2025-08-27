import streamlit as st
from openai import OpenAI
import tempfile
import os
from pathlib import Path
import time
from dotenv import load_dotenv
import io
from pydub import AudioSegment
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import threading
import queue

# Load environment variables
load_dotenv()

# Configuration - SET YOUR API KEY HERE
OPENAI_API_KEY = "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"  # Replace with your actual API key
WHISPER_MODEL = "whisper-1"  # Default Whisper model
GPT_MODEL = "gpt-4o-mini"  # Options: "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"
DEFAULT_SAMPLE_RATE = 16000  # Default sample rate for audio recording

# Page configuration
st.set_page_config(
    page_title="Voice to Text Translator & Summarizer",
    page_icon="🎙️",
    layout="wide"
)

# Custom CSS for dark theme UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86C1;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .feature-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: rgba(49, 51, 63, 0.5);
        border: 1px solid rgba(102, 126, 234, 0.2);
        margin: 1rem 0;
    }
    .result-box {
        padding: 1rem;
        background: rgba(49, 51, 63, 0.7);
        border-radius: 8px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        margin: 1rem 0;
        color: #e6e6e6;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        opacity: 0.9;
    }
    /* Dark theme for text areas and inputs */
    .stTextArea textarea {
        background-color: rgba(49, 51, 63, 0.7) !important;
        color: #e6e6e6 !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(49, 51, 63, 0.7) !important;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>🎙️ Voice to Text Translator & Summarizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Convert speech to text, translate, and summarize using OpenAI's Whisper and GPT models</p>", unsafe_allow_html=True)

# Initialize session state
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = ""
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = []
if 'sample_rate' not in st.session_state:
    st.session_state.sample_rate = DEFAULT_SAMPLE_RATE

# Initialize OpenAI client with hardcoded API key
try:
    # Clear any potential proxy settings that might interfere
    for key in list(os.environ.keys()):
        if 'proxy' in key.lower() or 'PROXY' in key:
            del os.environ[key]
    
    # Try to get API key from environment first, then use hardcoded
    api_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
    
    if api_key and api_key != "your_openai_api_key_here":
        os.environ["OPENAI_API_KEY"] = api_key
        client = OpenAI(api_key=api_key)
    else:
        st.error("⚠️ Please set your OpenAI API key in the app.py file (line 20)")
        st.info("Replace 'your_openai_api_key_here' with your actual OpenAI API key")
        client = None
except Exception as e:
    st.error(f"Error initializing OpenAI client: {str(e)}")
    client = None

# Sidebar for language configuration only
with st.sidebar:
    st.header("🌐 Audio Settings")
    
    source_language = st.selectbox(
        "Source Language (for transcription)",
        ["auto-detect", "en", "es", "fr", "de", "hi", "kn", "ta", "te", "ml"],
        help="Select the source language or auto-detect"
    )
    
    st.divider()
    
    # # Troubleshooting section
    # st.header("🔧 Troubleshooting")
    # st.markdown("""
    # **If you get errors:**
    
    # 🎵 **Audio Issues:**
    # • Use WAV, MP3, or M4A files
    # • Keep files under 25MB
    # • Ensure audio is not silent
    
    # 🔑 **API Issues:**
    # • Check your API key is correct
    # • Verify you have OpenAI credits
    # • Try shorter audio files
    
    # 🎤 **Recording Issues:**
    # • Allow microphone permissions
    # • Speak clearly and loudly
    # • Try shorter recordings first
    # """)
    
    st.divider()
    
    # st.header("💡 Instructions Help")
    # st.markdown("""
    # **Example Instructions:**
    # - "Translate this to English"
    # - "Summarize this in Kannada" 
    # - "Give me bullet points in Hindi"
    # - "Translate to Tamil and summarize"
    # - "Extract key points in English"
    # - "Convert to Spanish with main ideas"
    # """)
    
    # st.divider()
    
    # st.header("📋 Supported Languages")
    # st.markdown("""
    # - **English** - en
    # - **Kannada** - kn  
    # - **Hindi** - hi
    # - **Tamil** - ta
    # - **Telugu** - te
    # - **Malayalam** - ml
    # - **Spanish** - es
    # - **French** - fr
    # - **German** - de
    # """)

# Audio recording functions
def record_audio_chunk(q, sample_rate, duration=30):
    """Record audio in chunks"""
    def callback(indata, frames, time, status):
        q.put(indata.copy())
    
    with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback):
        sd.sleep(int(duration * 1000))

def save_audio_to_file(audio_data, sample_rate):
    """Save recorded audio to a WAV file"""
    try:
        # Check if we have audio data
        if not audio_data or len(audio_data) == 0:
            raise ValueError("No audio data to save")
        
        # Concatenate audio chunks
        audio_array = np.concatenate(audio_data, axis=0)
        
        # Ensure audio array is properly shaped (remove extra dimensions)
        if len(audio_array.shape) > 1:
            audio_array = audio_array.squeeze()
        
        # Normalize and convert to int16
        # Ensure values are in the correct range
        audio_array = np.clip(audio_array, -1.0, 1.0)
        audio_array = (audio_array * 32767).astype(np.int16)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            # Use scipy to write WAV file for better compatibility
            wavfile.write(tmp_file.name, sample_rate, audio_array)
            return tmp_file.name
            
    except Exception as e:
        st.error(f"Error saving audio file: {str(e)}")
        raise

# Main content area
if client:
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📁 Upload Audio", "🎤 Record Audio"])
    
    with tab1:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.subheader("Upload Audio File")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'webm', 'ogg'],
            help="Supported formats: WAV, MP3, MP4, MPEG, MPGA, M4A, WEBM, OGG"
        )
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav')
            
            if st.button("🎯 Process Uploaded Audio", key="process_upload"):
                with st.spinner("Processing audio..."):
                    try:
                        # Reset file pointer
                        uploaded_file.seek(0)
                        
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.name).suffix, delete=False) as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            audio_file_path = tmp_file.name
                        
                        # Always try to convert to WAV for better compatibility
                        file_extension = Path(uploaded_file.name).suffix.lower()
                        
                        try:
                            # Load audio with pydub and convert to WAV with optimal settings
                            audio = AudioSegment.from_file(audio_file_path)
                            
                            # Optimize audio settings for Whisper
                            audio = audio.set_frame_rate(16000)  # 16kHz for speech
                            audio = audio.set_channels(1)  # Mono
                            
                            # Create new WAV file
                            converted_path = audio_file_path.replace(file_extension, '.wav')
                            audio.export(converted_path, format='wav')
                            
                            # Clean up original temp file if different
                            if converted_path != audio_file_path:
                                os.unlink(audio_file_path)
                                audio_file_path = converted_path
                            
                            st.info(f"✅ Audio converted to optimized WAV format ({audio.frame_rate}Hz, {audio.channels} channel)")
                            
                        except Exception as conv_error:
                            st.warning(f"Could not optimize audio format: {conv_error}")
                            st.info("Continuing with original file format...")
                        
                        # Validate file before transcription
                        if not os.path.exists(audio_file_path):
                            raise ValueError("Audio file not found")
                        
                        file_size = os.path.getsize(audio_file_path)
                        if file_size == 0:
                            raise ValueError("Audio file is empty")
                        
                        if file_size > 25 * 1024 * 1024:  # 25MB limit
                            raise ValueError("Audio file is too large. OpenAI has a 25MB limit.")
                        
                        # Transcribe audio
                        with open(audio_file_path, 'rb') as audio_file:
                            try:
                                if source_language == "auto-detect":
                                    transcript = client.audio.transcriptions.create(
                                        model=WHISPER_MODEL,
                                        file=audio_file,
                                        response_format="text"
                                    )
                                else:
                                    transcript = client.audio.transcriptions.create(
                                        model=WHISPER_MODEL,
                                        file=audio_file,
                                        language=source_language,
                                        response_format="text"
                                    )
                            except Exception as api_error:
                                # Try again with a simpler request
                                st.warning("First attempt failed, trying simplified request...")
                                audio_file.seek(0)  # Reset file pointer
                                transcript = client.audio.transcriptions.create(
                                    model=WHISPER_MODEL,
                                    file=audio_file
                                )
                        
                        st.session_state.transcribed_text = transcript
                        
                        # Clean up temp file
                        if os.path.exists(audio_file_path):
                            os.unlink(audio_file_path)
                        
                        st.success("✅ Audio processed successfully!")
                        
                    except Exception as e:
                        error_msg = str(e)
                        if "invalid_request_error" in error_msg:
                            st.error("❌ Error processing audio file. This could be due to:")
                            st.info("• File format not supported by OpenAI\n• Audio file corrupted\n• File too large (>25MB)\n• Network connectivity issues")
                            st.info("💡 Try converting your audio to WAV format or use a shorter recording")
                        elif "401" in error_msg or "authentication" in error_msg.lower():
                            st.error("❌ Authentication error: Please check your OpenAI API key")
                        elif "insufficient_quota" in error_msg:
                            st.error("❌ API quota exceeded: Please check your OpenAI account credits")
                        else:
                            st.error(f"❌ Error processing audio: {error_msg}")
                        
                        # Clean up any remaining temp files
                        if 'audio_file_path' in locals() and os.path.exists(audio_file_path):
                            os.unlink(audio_file_path)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.subheader("Record Audio")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            duration = st.number_input(
                "Recording Duration (seconds)",
                min_value=1,
                max_value=60,
                value=10,
                step=1
            )
        
        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            if st.button("🔴 Start Recording", key="start_rec"):
                st.session_state.recording = True
                st.session_state.audio_data = []
        
        if st.session_state.recording:
            with st.spinner(f"Recording for {duration} seconds..."):
                # Record audio
                audio_data = []
                
                def audio_callback(indata, frames, time, status):
                    if status:
                        st.warning(f"Audio recording status: {status}")
                    audio_data.append(indata.copy())
                
                # Record audio
                try:
                    # Check if microphone is available
                    devices = sd.query_devices()
                    input_device = sd.default.device[0]
                    if input_device is None:
                        raise ValueError("No input audio device found. Please check your microphone.")
                    
                    with sd.InputStream(samplerate=st.session_state.sample_rate, 
                                      channels=1, 
                                      callback=audio_callback,
                                      dtype='float32'):
                        sd.sleep(int(duration * 1000))
                    
                    st.session_state.audio_data = audio_data
                    st.session_state.recording = False
                    
                    # Check if we got audio data
                    if not audio_data:
                        raise ValueError("No audio was recorded. Please check your microphone permissions.")
                    
                    # Save and process the recorded audio
                    audio_file_path = save_audio_to_file(audio_data, st.session_state.sample_rate)
                    
                    # Verify file exists and has content
                    if not os.path.exists(audio_file_path):
                        raise ValueError("Failed to save audio file")
                    
                    file_size = os.path.getsize(audio_file_path)
                    if file_size < 1000:  # Less than 1KB probably means empty file
                        raise ValueError("Audio file is too small. Please try recording again.")
                    
                    # Play back the recorded audio
                    st.audio(audio_file_path, format='audio/wav')
                    
                    # Validate recorded file
                    file_size = os.path.getsize(audio_file_path)
                    if file_size > 25 * 1024 * 1024:  # 25MB limit
                        raise ValueError("Recorded audio is too large. Try shorter recording duration.")
                    
                    # Transcribe audio
                    with open(audio_file_path, 'rb') as audio_file:
                        try:
                            if source_language == "auto-detect":
                                transcript = client.audio.transcriptions.create(
                                    model=WHISPER_MODEL,
                                    file=audio_file,
                                    response_format="text"
                                )
                            else:
                                transcript = client.audio.transcriptions.create(
                                    model=WHISPER_MODEL,
                                    file=audio_file,
                                    language=source_language,
                                    response_format="text"
                                )
                        except Exception as api_error:
                            # Try again with a simpler request
                            st.warning("First attempt failed, trying simplified request...")
                            audio_file.seek(0)  # Reset file pointer
                            transcript = client.audio.transcriptions.create(
                                model=WHISPER_MODEL,
                                file=audio_file
                            )
                    
                    st.session_state.transcribed_text = transcript
                    
                    # Clean up temp file
                    if os.path.exists(audio_file_path):
                        os.unlink(audio_file_path)
                    
                    st.success("✅ Recording completed and processed!")
                    
                except sd.PortAudioError as e:
                    st.error(f"Audio device error: {str(e)}")
                    st.info("Please ensure your microphone is connected and permissions are granted.")
                    st.session_state.recording = False
                except Exception as e:
                    error_msg = str(e)
                    if "invalid_request_error" in error_msg:
                        st.error("❌ Error processing recorded audio. This could be due to:")
                        st.info("• Recording quality too low\n• Recording too short or silent\n• Network connectivity issues")
                        st.info("💡 Try recording again with clear speech")
                    elif "401" in error_msg or "authentication" in error_msg.lower():
                        st.error("❌ Authentication error: Please check your OpenAI API key")
                    elif "insufficient_quota" in error_msg:
                        st.error("❌ API quota exceeded: Please check your OpenAI account credits")
                    else:
                        st.error(f"❌ Error recording audio: {error_msg}")
                    
                    st.session_state.recording = False
                    # Clean up any remaining temp files
                    if 'audio_file_path' in locals() and os.path.exists(audio_file_path):
                        os.unlink(audio_file_path)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Display transcribed text
    if st.session_state.transcribed_text:
        st.divider()
        st.subheader("📝 Transcribed Text")
        
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.text_area(
            "Original Transcription",
            value=st.session_state.transcribed_text,
            height=150,
            key="transcription_display"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Processing options
        st.divider()
        st.subheader("🔧 Processing Instructions")
        
        # Custom prompt input - always visible
        st.markdown("**📝 Enter your processing instructions:**")
        
        custom_prompt = st.text_area(
            "Processing Instructions",
            placeholder="E.g., 'Translate this audio to English', 'Summarize this in Kannada', 'Extract key points and translate to Hindi', 'Give me bullet points in English'",
            height=120,
            help="Write specific instructions for how you want the audio to be processed. Be clear about the language and format you want.",
            key="processing_instructions",
            label_visibility="collapsed"
        )
        
        # Quick instruction buttons
        st.markdown("**⚡ Quick Instructions:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.button("🌐 Translate to English", disabled=True, help="Copy: Translate this to English")
        
        with col2:
            st.button("🔍 Summarize in English", disabled=True, help="Copy: Summarize this in English")
        
        with col3:
            st.button("🌐 Translate to Kannada", disabled=True, help="Copy: Translate this to Kannada")
        
        with col4:
            st.button("🔍 Summarize in Kannada", disabled=True, help="Copy: Summarize this in Kannada")
        
        # Show example instructions instead of more buttons
        st.markdown("**💡 More Example Instructions:**")
        st.info("""
        **Try these instructions:**
        - "Translate this to Hindi"
        - "Give me bullet points in Tamil"  
        - "Summarize in Telugu with main points"
        - "Convert to Spanish and make it brief"
        - "Extract key information in Malayalam"
        - "Translate to French and summarize"
        """)
        
        st.markdown("---")
        
        # Process button
        if st.button("🚀 Process Text", key="process_text"):
            if not custom_prompt or custom_prompt.strip() == "":
                st.error("⚠️ Please enter processing instructions before clicking Process Text!")
            else:
                with st.spinner("Processing your request..."):
                    try:
                        # Use the custom prompt directly
                        prompt = f"""
                        {custom_prompt}
                        
                        Audio Content: {st.session_state.transcribed_text}
                        
                        Please follow the instructions above and process the audio content accordingly.
                        """
                        
                        # Call GPT model
                        response = client.chat.completions.create(
                            model=GPT_MODEL,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant that specializes in translation, summarization, and text processing. Follow the user's instructions carefully and provide accurate, high-quality output."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.7,
                            max_tokens=1500
                        )
                        
                        result = response.choices[0].message.content
                        
                        # Display result
                        st.divider()
                        st.subheader("📊 Processing Result")
                        
                        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                        st.markdown("**Your Instructions:**")
                        st.info(custom_prompt)
                        st.markdown("**Result:**")
                        st.write(result)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Copy result
                        st.markdown("**📋 Copy Result:**")
                        st.code(result, language=None)
                        
                        st.success("✅ Processing completed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error processing text: {str(e)}")
                        st.info("💡 Try simplifying your instructions or check your internet connection.")
else:
    st.warning("⚠️ Please enter your OpenAI API key in the sidebar to start using the application.")
    st.info("💡 You can get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>Built with ❤️ using Streamlit and OpenAI</p>
    <p style='font-size: 0.9rem;'>Powered by Whisper & GPT Models</p>
</div>
""", unsafe_allow_html=True)
