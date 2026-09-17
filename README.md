# 🎬 VidSynth AI

> Turn any YouTube video into structured notes, key topics, and an AI chatbot — without watching the whole thing.

---

## The Problem

You open a 2-hour lecture. You need one concept from it.
You scrub through. You miss it. You rewatch. You take notes while watching — and miss more.

Most people waste hours just *finding* what they need inside a video.

**VidSynth AI solves this.** Paste a YouTube link. In a minute, you get clear notes written like a topper student wrote them, the 5 most important topics, and a chatbot you can ask anything about the video.

---

## What It Does

| Feature | How it helps |
|---|---|
| **Topper-style notes** | AI reads the transcript and writes notes the way a good student would — with explanations, examples, and tips |
| **5 key topics** | Quick overview of what the video actually covers |
| **Chat with video** | Ask anything — "what did he say about X?" — get a direct answer |
| **Paste transcript** | No YouTube link? Paste any transcript directly |
| **Multi-language** | Works with Hindi, Urdu, Spanish, French, and more — auto-translates to English |
| **Long video support** | Splits long videos into chunks, processes each part, stitches results together |
| **PDF download** | Save your notes as a clean PDF |

---

## Tech Stack

| What | Tool | Why |
|---|---|---|
| UI | Streamlit | Runs in the browser, no frontend code needed |
| Primary AI | Groq — `qwen/qwen3-32b` | Fast inference, generous free tier |
| Fallback AI 1 | Gemini 2.5 Flash Lite | Reliable when Groq hits limits |
| Fallback AI 2 | OpenRouter — Gemma 4 | Third safety net |
| Transcript | youtube-transcript-api | Pulls captions directly from YouTube |
| Audio fallback | faster-whisper + yt-dlp | Transcribes audio when captions are off |
| Notes pipeline | LangChain | Handles chunking, prompt chaining, long videos |
| Chat (RAG) | Chroma + HuggingFace Embeddings | Finds the right parts of the video for each question |
| PDF | FPDF2 | Generates downloadable notes |
| Config | python-dotenv | Keeps API keys out of code |

---

## How It Works

```
YouTube URL  ──►  Fetch transcript (captions or audio)
                        │
                  Translate to English (if needed)
                        │
              Split into chunks (long videos only)
                        │
              AI writes notes + extracts topics
                        │
         Read notes  /  Chat with video  /  Download PDF
```

---

## Project Structure

```
YouTube-Notes/
├── app.py                   # All UI — sidebar, buttons, pages
├── supporting_functions.py  # All AI logic — transcript, notes, chat, PDF
├── requirements.txt         # Python packages
├── .env                     # Your API keys (never commit this)
└── README.md
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/your-username/YouTube-Notes.git
cd YouTube-Notes
pip install -r requirements.txt
```

### 2. Create your `.env` file

```env
GOOGLE_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
```

Get free keys from:
- **Gemini** → [aistudio.google.com](https://aistudio.google.com)
- **Groq** → [console.groq.com](https://console.groq.com)
- **OpenRouter** → [openrouter.ai](https://openrouter.ai)

### 3. Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## How to Use

**Option A — YouTube URL**
1. Paste a YouTube link in the sidebar
2. Set the language code (`en`, `hi`, `ur`, etc.)
3. Pick **Notes For You** or **Chat with Video**
4. Click **Start Processing**

**Option B — Paste Transcript**
1. Switch input to **Paste Transcript**
2. Paste your transcript text
3. Pick your task and click **Start Processing**

---

## AI Fallback System

The app never silently crashes on quota limits. It tries providers in this order:

```
Groq  →  Gemini  →  OpenRouter
```

If all three hit their limits, you get a clear error message explaining what happened and when to retry.

---

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Gemini AI (fallback) |
| `GROQ_API_KEY` | Recommended | Primary AI provider |
| `OPENROUTER_API_KEY` | Optional | Third fallback |

---

## Known Limits

- Videos with disabled captions use local audio transcription — slower but still works
- Very long videos (2+ hours) take more time and more API calls
- Free API tiers have per-minute limits — if you hit them, wait 1-2 minutes and retry

---

*Built for students and professionals who can't afford to rewatch an entire lecture just to find one thing.*