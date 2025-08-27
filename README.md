# 🎙️ Voice to Text Translator & Summarizer

A powerful Streamlit web application that converts voice to text using OpenAI's Whisper model, with capabilities for translation and summarization in multiple languages including English, Kannada, Hindi, and more.

## ✨ Features

- **🎤 Voice Recording**: Record audio directly from your microphone
- **📁 File Upload**: Upload audio files in various formats (WAV, MP3, MP4, etc.)
- **🔊 Speech-to-Text**: Convert audio to text using OpenAI's Whisper model
- **🌐 Multi-language Translation**: Translate text to multiple languages including:
  - English, Kannada, Hindi, Spanish, French, German, Tamil, Telugu, Malayalam
- **📝 Text Summarization**: Create concise summaries of transcribed text
- **🎯 Custom Prompts**: Use custom prompts for specific processing needs
- **💡 Combined Operations**: Translate and summarize in a single operation

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API Key ([Get it here](https://platform.openai.com/api-keys))
- FFmpeg (for audio processing)

### Step 1: Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [FFmpeg website](https://ffmpeg.org/download.html)

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure API Key

1. Open `app.py` file
2. Find line 20: `OPENAI_API_KEY = "your_openai_api_key_here"`
3. Replace `"your_openai_api_key_here"` with your actual OpenAI API key
4. Save the file

## 🚀 Usage

### Starting the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Application

1. **Configure Audio Settings**:
   - Select source language for transcription in the sidebar (auto-detect works well)
   - Review helpful examples and supported languages in the sidebar

2. **Input Audio**:
   - **Upload**: Click on "Upload Audio" tab and select an audio file
   - **Record**: Click on "Record Audio" tab, set duration, and click "Start Recording"

3. **Write Processing Instructions**:
   - Once audio is transcribed, write specific instructions in the text area
   - Examples: "Translate this to English", "Summarize in Kannada", "Give bullet points in Hindi"
   - Use the quick instruction buttons for common tasks
   - Be specific about language and format you want

4. **Process and Get Results**:
   - Click "Process Text" to generate results
   - View your custom-processed output
   - Copy results using the provided code block

See [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) for detailed examples and tips.

## 📋 Supported Audio Formats

- WAV, MP3, MP4, MPEG, MPGA, M4A, WEBM, OGG

## 🎯 Use Cases

### 1. Simple Translation
```
Upload or record audio → Write: "Translate this to English" → Process
```

### 2. Summarization
```
Upload or record audio → Write: "Summarize this in Kannada with key points" → Process  
```

### 3. Formatted Output
```
Upload or record audio → Write: "Give me bullet points in Hindi from this audio" → Process
```

### 4. Complex Instructions
```
Upload or record audio → Write: "Translate to Tamil and create an executive summary" → Process
```

### 5. Meeting Notes
```
Upload meeting audio → Write: "Extract action items and decisions in English as numbered list" → Process
```

## 🔧 Configuration Options

### Models (Hardcoded in app.py)
- **Whisper Model**: whisper-1 (OpenAI's speech recognition model)
- **GPT Model**: gpt-4o-mini (can be changed to gpt-4o or gpt-3.5-turbo in code)
- **Sample Rate**: 16000 Hz (optimized for speech)

### Languages
- **Source Languages**: Auto-detect or specify (English, Kannada, Hindi, etc.)
- **Target Languages**: Multiple selection for translation output

## 🐛 Troubleshooting

### Test Your Setup

Run the audio test script to diagnose issues:
```bash
python test_audio.py
```

### Common Issues

1. **API Key not working**:
   - Make sure you've replaced the API key in line 20 of `app.py`
   - Verify your API key is correct and has sufficient credits

2. **"Invalid file format" error**:
   - Ensure FFmpeg is properly installed
   - Try with a different audio file format (WAV recommended)

3. **Audio recording not working**:
   - Check microphone permissions
   - Run `python test_audio.py` to diagnose issues

4. **Installation issues**:
   - Make sure Python 3.8+ is installed
   - Try creating a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     pip install -r requirements.txt
     ```

## 📝 Notes

- **Recording Duration**: Up to 60 seconds for optimal performance
- **File Size**: Large audio files may take longer to process
- **API Costs**: Using this application will incur OpenAI API charges
- **Privacy**: Audio files are temporarily stored during processing and deleted afterward

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is open source and available under the MIT License.

---

**Made with ❤️ using Streamlit and OpenAI**
