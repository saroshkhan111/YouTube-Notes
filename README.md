# 🎬 YouTube RAG Agent

Transform any YouTube video into structured notes, key topics, or an interactive chatbot using AI. Built with Streamlit, LangChain, and Google Gemini.

## ✨ Features

- **📝 Smart Notes Generation**: Automatically extract structured, concise notes from any YouTube video
- **🔍 Key Topics Extraction**: Get the 5 most important topics discussed in the video
- **💬 Chat with Video**: Ask questions about the video content using RAG (Retrieval Augmented Generation)
- **🌍 Multi-Language Support**: Process videos in any language (auto-translates to English)
- **🆓 100% Free**: Uses local embeddings (HuggingFace) - no API costs for vector storage
- **🔒 Privacy First**: All embeddings are created locally on your machine

## 🚀 Quick Start

### Prerequisites

- Python 3.12 or higher
- Google Gemini API key (free tier available)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd youtube
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root directory:
   ```bash
   touch .env
   ```
   
   Add your Google Gemini API key to the `.env` file:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```
   
   **How to get your API key:**
   - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Sign in with your Google account
   - Click "Create API Key"
   - Copy the key and paste it in your `.env` file

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

The app will automatically open in your browser at `http://localhost:8501`

## 📖 How to Use

### 1. Generate Notes

1. Paste a YouTube URL in the sidebar
2. Enter the video language code (e.g., `en` for English, `hi` for Hindi, `es` for Spanish)
3. Select "Notes For You"
4. Click "✨ Start Processing"
5. Wait for the AI to extract topics and generate notes

### 2. Chat with Video

1. Paste a YouTube URL in the sidebar
2. Enter the video language code
3. Select "Chat with Video"
4. Click "✨ Start Processing"
5. Once processing is complete, ask questions about the video content in the chat interface

## ⚙️ Configuration

### Language Codes

Use ISO 639-1 language codes:
- English: `en`
- Hindi: `hi`
- Spanish: `es`
- French: `fr`
- German: `de`
- Japanese: `ja`
- Korean: `ko`
- [Full list of language codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)

### GPU Support (Optional)

If you have an NVIDIA GPU, you can speed up embedding generation:

1. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. Edit `supporting_functions.py` line 165:
   ```python
   model_kwargs={'device': 'cuda'}  # Change from 'cpu' to 'cuda'
   ```

## ⚠️ Important Notes & Tips

### First Run Will Be Slow
- **First time only**: The app downloads the `all-MiniLM-L6-v2` embedding model (~80MB)
- This happens automatically and only once
- The model is cached locally for future use
- Subsequent runs will be much faster

### Local Processing
- **Embeddings are created locally** using HuggingFace models
- No external API calls for vector storage (completely free!)
- Only Google Gemini API is used for text generation (translation, notes, chat responses)
- Your data stays on your machine during embedding creation

### Rate Limits
- Google Gemini free tier has rate limits for text generation
- If you hit limits, wait a few minutes and try again
- The embedding model has **no rate limits** since it runs locally

### Video Length
- Very long videos (3+ hours) may take several minutes to process
- Processing time depends on video length and your hardware

### Memory Usage
- Local embedding model uses ~500MB RAM
- Vector store size depends on video length
- Recommended: At least 4GB RAM available

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Google Gemini 2.5 Flash Lite
- **Embeddings**: HuggingFace Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB
- **Framework**: LangChain
- **Transcription**: YouTube Transcript API

## 📁 Project Structure

```
youtube/
├── app.py                    # Main Streamlit application
├── supporting_functions.py   # Core functionality (transcript, embeddings, RAG)
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Project configuration
├── .env                     # Environment variables (API keys) - CREATE THIS
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🐛 Troubleshooting

### "No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

### "Error fetching transcript"
- Check if the video has captions/subtitles available
- Verify the language code is correct
- Try a different video

### "Quota exceeded" for Google Gemini
- You've hit the free tier rate limit
- Wait a few minutes or upgrade your API plan
- The embedding creation won't be affected (runs locally)

### "Failed to create vector store"
- Ensure you have enough disk space (~500MB)
- Check if `chromadb` is installed: `pip install chromadb`
- Restart the application

### Slow Performance
- First run downloads the embedding model (one-time ~80MB)
- Close other applications to free up RAM
- Consider using GPU acceleration (see Configuration section)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests


## 🙏 Acknowledgments

- Google Gemini for powerful LLM capabilities
- HuggingFace for free local embeddings
- Streamlit for the amazing web framework
- LangChain for RAG implementation

## 📧 Support

If you encounter any issues or have questions:
1. Check the Troubleshooting section above
2. Search for existing issues on GitHub
3. Create a new issue with detailed information

---

**Made using  Open Source tools**
