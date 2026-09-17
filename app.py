import streamlit as st
from streamlit import spinner

from supporting_functions import (
    LLMQuotaError,
    create_chunks,
    create_vector_store,
    extract_video_id,
    generate_notes_and_topics,
    generate_notes_from_transcript,
    get_transcript,
    rag_answer,
    transcribe_video_audio,
    translate_transcript,
)

# --- Streamlit Page Config ---
st.set_page_config(page_title="VidSynth AI", page_icon="🎬", layout="wide")

st.html(
    """
    <style>
    @media print {
        @page { margin: 16mm; }

        [data-testid="stSidebar"],
        [data-testid="stToolbar"],
        [data-testid="stStatusWidget"],
        [data-testid="stDecoration"],
        [data-testid="stBottom"],
        [data-testid="stChatInput"],
        [data-testid="stSpinner"],
        header,
        .stButton {
            display: none !important;
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {
            background: #ffffff !important;
            color: #111827 !important;
        }

        [data-testid="stMainBlockContainer"] {
            max-width: none !important;
            padding: 0 !important;
        }

        h1, h2, h3, p, li, span, div {
            color: #111827 !important;
            opacity: 1 !important;
        }
    }
    </style>
    """
)

# Initialize Session State Variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "full_transcript" not in st.session_state:
    st.session_state.full_transcript = None

# --- Sidebar (Inputs) ---
with st.sidebar:
    st.title("🎬 VidSynth AI")
    st.markdown("---")
    st.markdown(
        "Transform any YouTube video into key topics, notes, or a chatbot."
    )
    st.markdown("### Input Details")

    # --- NEW: Input Method selector ---
    input_method = st.radio(
        "Input Method",
        ["YouTube URL", "Paste Transcript"],
        horizontal=True,
    )

    # Show YouTube URL + language only when "YouTube URL" is selected
    if input_method == "YouTube URL":
        youtube_url = st.text_input(
            "YouTube URL", placeholder="https://www.youtube.com/watch?v=..."
        )
        language = st.text_input(
            "Video Language Code",
            placeholder="e.g., en, hi, es, fr",
            value="en",
        )
    else:
        # Hide URL/language fields; show transcript text area instead
        youtube_url = ""
        language = "en"
        pasted_transcript = st.text_area(
            "Paste your transcript here",
            placeholder="Copy and paste the full transcript of the video...",
            height=250,
        )

    task_option = st.radio(
        "Choose what you want to generate:",
        ["Chat with Video", "Notes For You"],
    )

    submit_button = st.button("✨ Start Processing")
    st.markdown("---")

# --- Main Page ---
st.title("YouTube Content Synthesizer")
st.markdown("Paste a video link or transcript and select a task from the sidebar.")

# --- Processing Flow ---
if submit_button:

    # ── PATH A: YouTube URL ──────────────────────────────────────────────────
    if input_method == "YouTube URL":
        if youtube_url and language:
            video_id = extract_video_id(youtube_url)
            if video_id:
                with spinner("Step 1/3 : Fetching Transcript....."):
                    transcript = get_transcript(video_id, language)

                if not transcript:
                    with spinner("No captions found. Transcribing audio..."):
                        transcript = transcribe_video_audio(video_id, language)

                if transcript:
                    if language != "en":
                        with spinner(
                            "Step 1.5/3 : Translating Transcript into English..."
                        ):
                            transcript = translate_transcript(transcript)

                    st.session_state.full_transcript = transcript

                    if task_option == "Notes For You":
                        progress_bar = st.progress(0)
                        progress_status = st.empty()

                        def update_progress(current: int, total: int) -> None:
                            progress_bar.progress(current / total)
                            progress_status.caption(
                                f"Processing video section {current} of {total}..."
                            )

                        try:
                            with spinner("Creating topics and notes..."):
                                import_topics, notes = generate_notes_and_topics(
                                    transcript, update_progress
                                )
                        except LLMQuotaError as error:
                            progress_bar.empty()
                            progress_status.empty()
                            st.error(str(error))
                        else:
                            st.subheader("Important Topics")
                            st.write(import_topics)
                            st.markdown("---")
                            st.subheader("Notes for you")
                            st.write(notes)
                            progress_bar.empty()
                            progress_status.empty()
                            st.success("Summary and Notes Generated.")

                    if task_option == "Chat with Video":
                        with st.spinner(
                            "Step 2/3: Creating chunks and vector store...."
                        ):
                            chunks = create_chunks(transcript)
                            vectorstore = create_vector_store(chunks)

                            if vectorstore is not None:
                                st.session_state.vector_store = vectorstore
                                st.session_state.messages = []
                                st.success(
                                    "Video is ready for chat! Ask anything below."
                                )
                            else:
                                st.error(
                                    "Failed to create vector store. Please try again."
                                )
                else:
                    st.error(
                        "Failed to fetch transcript. Please check the video ID or language."
                    )
        else:
            st.warning("Please enter both YouTube URL and language code.")

    # ── PATH B: Paste Transcript ─────────────────────────────────────────────
    else:
        if pasted_transcript and pasted_transcript.strip():
            transcript = pasted_transcript.strip()
            st.session_state.full_transcript = transcript

            if task_option == "Notes For You":
                progress_bar = st.progress(0)
                progress_status = st.empty()

                def update_progress(current: int, total: int) -> None:
                    progress_bar.progress(current / total)
                    progress_status.caption(
                        f"Processing section {current} of {total}..."
                    )

                try:
                    with spinner("Creating topics and notes from your transcript..."):
                        import_topics, notes = generate_notes_from_transcript(
                            transcript, update_progress
                        )
                except LLMQuotaError as error:
                    progress_bar.empty()
                    progress_status.empty()
                    st.error(str(error))
                else:
                    st.subheader("Important Topics")
                    st.write(import_topics)
                    st.markdown("---")
                    st.subheader("Notes for you")
                    st.write(notes)
                    progress_bar.empty()
                    progress_status.empty()
                    st.success("Notes Generated from your transcript.")

            if task_option == "Chat with Video":
                with st.spinner("Creating chunks and vector store..."):
                    chunks = create_chunks(transcript)
                    vectorstore = create_vector_store(chunks)

                    if vectorstore is not None:
                        st.session_state.vector_store = vectorstore
                        st.session_state.messages = []
                        st.success(
                            "Transcript is ready for chat! Ask anything below."
                        )
                    else:
                        st.error(
                            "Failed to create vector store. Please try again."
                        )
        else:
            st.warning("Please paste a transcript before clicking Start Processing.")


# --- Chatbot Interface ---
if task_option == "Chat with Video":
    st.divider()
    st.subheader("💬 Chat with Video")

    if st.session_state.vector_store is None:
        st.info(
            "👈 Enter a YouTube URL (or paste a transcript) and click **Start Processing** to begin chatting."
        )
    else:
        # Display history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # User input
        prompt = st.chat_input("Ask me anything about the video...")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"), st.spinner("Thinking..."):
                response = rag_answer(
                    prompt,
                    st.session_state.vector_store,
                    st.session_state.full_transcript,
                )
                st.write(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )