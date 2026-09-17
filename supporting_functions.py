import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    RequestBlocked,
    YouTubeTranscriptApiException,
)

load_dotenv()

SOURCE_CHUNK_SIZE = 20_000
SOURCE_CHUNK_OVERLAP = 500
FINAL_CONTEXT_SIZE = 24_000


class LLMQuotaError(Exception):
    pass


FALLBACK_ERRORS: list[tuple[str, str]] = []

QUOTA_ERROR_MARKERS = (
    "RESOURCE_EXHAUSTED",
    "429",
    "rate limit",
    "rate_limit",
    "quota",
    "free-models-per-day",
)


def shorten_error(message: str, limit: int = 300) -> str:
    message = " ".join(str(message).split())
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


def is_quota_error(message: str) -> bool:
    lowered = message.lower()
    return any(marker.lower() in lowered for marker in QUOTA_ERROR_MARKERS)


def build_quota_message(
    primary_message: str, fallback_errors: list[tuple[str, str]] | None = None
) -> str:
    parts = [
        (
            "AI request limit reached. Primary provider reported: "
            f"{shorten_error(primary_message)}"
        )
    ]
    if fallback_errors:
        for label, fallback_message in fallback_errors:
            parts.append(
                f"{label} fallback reported: {shorten_error(fallback_message)}"
            )
    else:
        parts.append("No fallback provider error was recorded.")
    parts.append(
        "Wait for the retry time shown by Gemini, check your OpenRouter and Groq "
        "limits, or try again after the daily quota resets."
    )
    return " ".join(parts)


@st.cache_resource
def get_llm():
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_google_genai import ChatGoogleGenerativeAI

    def make_error_recorder(label: str) -> BaseCallbackHandler:
        class FallbackErrorRecorder(BaseCallbackHandler):
            def on_llm_error(self, error, **kwargs) -> None:
                FALLBACK_ERRORS.append((label, str(error)))

        return FallbackErrorRecorder()

    gemini = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", temperature=0.3
    )
    openrouter_fallback = None
    groq_primary = None

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_api_key:
        try:
            from langchain_openai import ChatOpenAI
            openrouter_fallback = ChatOpenAI(
                model="google/gemma-4-26b-a4b-it:free",
                temperature=0.3,
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_api_key,
                callbacks=[make_error_recorder("OpenRouter")],
            )
        except ImportError:
            st.warning("langchain-openai is not installed, OpenRouter fallback is disabled.")
    else:
        st.warning(
            "OPENROUTER_API_KEY is missing. The OpenRouter fallback is disabled."
        )

    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            st.warning(
                "langchain-groq is not installed, the Groq fallback is disabled. "
                "Run `pip install langchain-groq` to enable it."
            )
        else:
            groq_primary = ChatGroq(
                model="qwen/qwen3.8-27b",
                temperature=0.3,
                api_key=groq_api_key,
                callbacks=[make_error_recorder("Groq")],
                max_retries=0,
                timeout=30,
            )
    else:
        st.warning("GROQ_API_KEY is missing. The Groq fallback is disabled.")

    if groq_primary:
        fallbacks = [gemini]
        if openrouter_fallback:
            fallbacks.append(openrouter_fallback)
        return groq_primary.with_fallbacks(fallbacks)

    if openrouter_fallback:
        return gemini.with_fallbacks([openrouter_fallback])

    st.warning("Running with Gemini only, no fallback provider is configured.")
    return gemini


def invoke_llm(chain, values: dict[str, str]) -> str:
    FALLBACK_ERRORS.clear()
    try:
        return str(chain.invoke(values).content)
    # Broad catch is intentional: provider errors differ across LangChain
    # integrations, and the error is re-raised unless it is a quota error.
    except Exception as error:
        message = str(error)
        fallback_errors: list[tuple[str, str]] = []
        for label, fallback_message in FALLBACK_ERRORS:
            if not any(existing == label for existing, _ in fallback_errors):
                fallback_errors.append((label, fallback_message))
        if is_quota_error(message) or any(
            is_quota_error(fallback_message) for _, fallback_message in fallback_errors
        ):
            raise LLMQuotaError(
                build_quota_message(message, fallback_errors)
            ) from error
        raise


def split_transcript_for_llm(transcript: str) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SOURCE_CHUNK_SIZE,
        chunk_overlap=SOURCE_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    return splitter.split_text(transcript)


def process_chunks(
    chunks: list[str], prompt_template: str, progress_callback=None
) -> list[str]:
    from langchain_core.prompts import ChatPromptTemplate

    chain = ChatPromptTemplate.from_template(prompt_template) | get_llm()
    results = []
    for index, chunk in enumerate(chunks, start=1):
        if progress_callback:
            progress_callback(index, len(chunks))
        results.append(invoke_llm(chain, {"transcript": chunk}))
    return results


def create_compact_overview(
    section_summaries: list[str], progress_callback=None
) -> str:
    combined_summaries = "\n\n".join(section_summaries)
    while len(combined_summaries) > FINAL_CONTEXT_SIZE:
        summary_chunks = split_transcript_for_llm(combined_summaries)
        section_summaries = process_chunks(
            summary_chunks,
            """
            Combine this portion of video notes into a compact factual overview.
            Keep the concepts, steps, examples, and important terms. Do not add facts.

            Video notes:
            {transcript}
            """,
            progress_callback,
        )
        combined_summaries = "\n\n".join(section_summaries)
    return combined_summaries


# 1. Extract video ID
def extract_video_id(url: str) -> str | None:
    match = re.search(
        r"(?:v=|\/|youtu\.be\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})", url
    )
    if match:
        return match.group(1)
    st.error("Invalid YouTube URL. Please enter a valid video link.")
    return None


# 2. Transcript Fetcher
def get_transcript(video_id: str, language: str = "en") -> str | None:
    try:
        languages = [language, "en", "en-US", "hi", "ur"]
        languages = list(dict.fromkeys(languages))

        raw_data = None

        try:
            ytt_api = YouTubeTranscriptApi()
            if hasattr(ytt_api, "fetch"):
                raw_data = ytt_api.fetch(video_id, languages=languages)
        except (AttributeError, RuntimeError, ValueError, RequestBlocked):
            pass

        if raw_data is None:
            try:
                if hasattr(YouTubeTranscriptApi, "get_transcript"):
                    raw_data = YouTubeTranscriptApi.get_transcript(
                        video_id, languages=languages
                    )
            except (AttributeError, RuntimeError, ValueError, RequestBlocked):
                pass

        if not raw_data:
            st.error(
                "Could not fetch transcript. Captions might be disabled on this video."
            )
            return None

        text_parts = []
        for item in raw_data:
            if isinstance(item, dict):
                text_parts.append(item.get("text", ""))
            elif hasattr(item, "text"):
                text_parts.append(getattr(item, "text", ""))
            elif isinstance(item, str):
                text_parts.append(item)

        full_transcript = " ".join(text_parts).strip()
        full_transcript = re.sub(r"\[.*?\]", "", full_transcript)
        full_transcript = re.sub(r"\(.*?\)", "", full_transcript)
        full_transcript = " ".join(full_transcript.split())

        if len(full_transcript.split()) < 15:
            st.warning("Warning: Transcript is very short or mostly empty.")

    except (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        YouTubeTranscriptApiException,
    ):
        st.warning("Could not fetch transcript. Falling back to audio transcription.")
        return None

    return full_transcript


@st.cache_resource
def get_transcription_model():
    from faster_whisper import WhisperModel

    return WhisperModel("base", device="cpu", compute_type="int8")


def transcribe_video_audio(video_id: str, language: str = "en") -> str | None:
    """Download a video's audio temporarily and transcribe it locally."""
    try:
        from av.error import FFmpegError
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import YoutubeDLError
    except ImportError as error:
        st.error(f"Could not transcribe the video audio: {error}")
        return None

    try:
        with tempfile.TemporaryDirectory() as temp_directory:
            output_template = str(Path(temp_directory) / "audio.%(ext)s")
            download_options = {
                "format": "bestaudio/best",
                "noplaylist": True,
                "outtmpl": output_template,
                "quiet": True,
            }
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            with YoutubeDL(download_options) as downloader:
                video_info = downloader.extract_info(video_url, download=True)
                audio_path = Path(downloader.prepare_filename(video_info))

            segments, _ = get_transcription_model().transcribe(
                str(audio_path), language=language or None, vad_filter=True
            )
            transcript = " ".join(segment.text.strip() for segment in segments)
            return transcript or None
    except (
        FFmpegError,
        YoutubeDLError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        st.error(f"Could not transcribe the video audio: {error}")
        return None


# 3. Translate
def translate_transcript(transcript: str, progress_callback=None) -> str:
    try:
        translation_prompt = """
        Translate this transcript into simple, clear English.
        Keep every detail. Do not summarize.

        Transcript:
        {transcript}
        """
        chunks = split_transcript_for_llm(transcript)
        if len(chunks) == 1:
            return process_chunks(chunks, translation_prompt)[0]
        return " ".join(process_chunks(chunks, translation_prompt, progress_callback))
    except (LLMQuotaError, RuntimeError, ValueError) as e:
        st.error(f"Error translating: {e}")
        return transcript


# 4. Important Topics (Simple Language)
def get_important_topics(transcript: str, progress_callback=None) -> str:
    try:
        from langchain_core.prompts import ChatPromptTemplate

        prompt = """
        You are a friendly teacher. Identify the 5 most critical core topics from this video transcript.

        Rules:
        - List exactly 5 numbered points (1. to 5.).
        - Each point must be ONE ultra-simple, crisp sentence explaining what is taught.
        - Use simple words so anyone can immediately understand the big picture.
        - No extra text or fluff.

        Transcript:
        {transcript}
        """
        chunks = split_transcript_for_llm(transcript)
        if len(chunks) == 1:
            return process_chunks(chunks, prompt)[0]

        candidates = process_chunks(
            chunks,
            """
            From this part of a video, extract the important concepts, steps,
            examples, and terms as concise factual bullet points. Do not add facts.

            Video section:
            {transcript}
            """,
            progress_callback,
        )
        overview = create_compact_overview(candidates, progress_callback)
        final_prompt = ChatPromptTemplate.from_template(prompt)
        chain = final_prompt | get_llm()
        return invoke_llm(chain, {"transcript": overview})
    except (LLMQuotaError, RuntimeError, ValueError) as e:
        st.error(f"Error generating topics: {e}")
        return "Could not generate topics."


# 5. Instructor-Style Detailed Notes (Spoon-Feeding & Token-Efficient Bullet Notes)
# This prompt is shared by generate_notes(), generate_notes_and_topics(),
# and generate_notes_from_transcript() so all paths get the same format.
NOTES_PROMPT = """
You are an expert tutor known for spoon-feeding difficult concepts to complete beginners and slow learners.
Turn this video transcript into clear, bite-sized, bullet-pointed study notes.

Goal:
1. Break down every concept step-by-step so simply that anyone understands on the first read.
2. Token-efficient: Use crisp bullet points, zero fluff, no lengthy paragraphs.

Format for each topic:
## [Topic Name]
- **Core Concept:** 1 direct, plain-English summary sentence.
- **Spoon-Fed Breakdown:**
  * Step 1 / Intuition in plain words.
  * Real-world analogy (e.g., "Think of it like...").
- **Key Takeaways:**
  * Crucial rule, fact, or workflow step.
  * Why it matters in practice.
- **Example / Code:** (if applicable) Short snippet with clear 1-line explanation.
- **Pro Tip / Pitfall:** 1 practical tip or common mistake to avoid.

Keep every bullet under 2 short sentences. Use bold keywords for quick scanning.

Transcript:
{transcript}
"""


def generate_notes(transcript: str, progress_callback=None) -> str:
    try:
        from langchain_core.prompts import ChatPromptTemplate

        prompt = NOTES_PROMPT
        chunks = split_transcript_for_llm(transcript)
        if len(chunks) == 1:
            return process_chunks(chunks, prompt)[0]

        section_summaries = process_chunks(
            chunks,
            """
            Create concise factual learning notes from this video section.
            Include the key ideas, steps, examples, and important words.
            Preserve the order of information. Do not add facts.

            Video section:
            {transcript}
            """,
            progress_callback,
        )
        overview = create_compact_overview(section_summaries, progress_callback)
        chain = ChatPromptTemplate.from_template(prompt) | get_llm()
        return invoke_llm(chain, {"transcript": overview})
    except (LLMQuotaError, RuntimeError, ValueError) as e:
        st.error(f"Error generating notes: {e}")
        return "Could not generate notes."


def generate_notes_and_topics(
    transcript: str, progress_callback=None
) -> tuple[str, str]:
    from langchain_core.prompts import ChatPromptTemplate

    chunks = split_transcript_for_llm(transcript)
    if len(chunks) == 1:
        return (
            get_important_topics(transcript),
            generate_notes(transcript),
        )

    section_summaries = process_chunks(
        chunks,
        """
        Create concise factual learning notes from this video section.
        Include the key ideas, steps, examples, and important words.
        Preserve the order of information. Do not add facts.

        Video section:
        {transcript}
        """,
        progress_callback,
    )
    overview = create_compact_overview(section_summaries, progress_callback)
    topics_chain = ChatPromptTemplate.from_template(
        """
        From these complete video notes, write exactly 5 important topics.
        Use simple everyday words, one short sentence per numbered topic.
        Do not add facts.

        Video notes:
        {transcript}
        """
    ) | get_llm()
    notes_chain = ChatPromptTemplate.from_template(NOTES_PROMPT) | get_llm()
    topics = invoke_llm(topics_chain, {"transcript": overview})
    notes = invoke_llm(notes_chain, {"transcript": overview})
    return topics, notes


# 6. NEW: Generate notes directly from a pasted transcript (no YouTube fetch)
def generate_notes_from_transcript(
    transcript: str, progress_callback=None
) -> tuple[str, str]:
    """
    Takes a raw transcript string (pasted by user) and returns (topics, notes).
    Works exactly like generate_notes_and_topics() but skips YouTube fetching.
    """
    return generate_notes_and_topics(transcript, progress_callback)


# 7. Create Chunks
def create_chunks(transcript: str):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents([transcript])


# 8. Vector Store
def create_vector_store(docs):
    try:
        from langchain_chroma import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        return Chroma.from_documents(
            docs, embedding, collection_name="yt_notes_chat"
        )
    except (RuntimeError, ValueError) as e:
        st.error(f"Error creating vector store: {e}")
        return None


# 9. RAG Answer
def rag_answer(
    question: str, vectorstore, full_transcript: str | None = None
) -> str:
    try:
        from langchain_core.prompts import ChatPromptTemplate

        context_text = ""
        q_lower = question.lower()

        broad_keywords = [
            "summarize",
            "summary",
            "points",
            "overview",
            "key point",
            "explain video",
            "main topic",
            "list",
            "notes",
        ]
        is_broad = any(kw in q_lower for kw in broad_keywords)

        if is_broad and full_transcript:
            context_text = full_transcript[:15000]
        elif vectorstore is not None:
            results = vectorstore.similarity_search(question, k=6)
            context_text = "\n\n".join([doc.page_content for doc in results])

        if not context_text and full_transcript:
            context_text = full_transcript[:15000]

        prompt = ChatPromptTemplate.from_template("""
        You are a kind and simple teacher.
        Answer in very easy words using the video context.

        If user asks for summary or points, give clear bullet points.
        Only say "I could not find this in the video" if it is truly not there.

        Context:
        {context}

        Question:
        {question}

        Easy Answer:
        """)

        chain = prompt | get_llm()
        return invoke_llm(chain, {"context": context_text, "question": question})
    except (LLMQuotaError, RuntimeError, ValueError) as e:
        return f"Error: {e}"


# 10. Create PDF from Notes
def create_pdf(notes_text: str, topics_text: str = "") -> bytes:
    """Generate a clean PDF from the notes and return bytes."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "VidSynth AI - Easy Notes", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(
        0,
        8,
        f"Generated on: {datetime.now(datetime.UTC).strftime('%d %b %Y, %I:%M %p')}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(6)

    def clean_text(text: str) -> str:
        text = re.sub(r"[^\x00-\x7F]+", "", text)
        text = text.replace("#", "").replace("*", "").replace("`", "")
        return text.strip()

    if topics_text:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Important Topics", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        for line in topics_text.split("\n"):
            line = clean_text(line)
            if line:
                pdf.multi_cell(0, 7, line)
        pdf.ln(6)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Detailed Easy Notes", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)

    for line in notes_text.split("\n"):
        line = clean_text(line)
        if not line:
            pdf.ln(3)
            continue

        if line.startswith("##") or line.isupper() or line.endswith(":"):
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 8, line.replace("##", "").strip())
            pdf.set_font("Helvetica", "", 11)
        else:
            pdf.multi_cell(0, 7, line)

    return bytes(pdf.output())