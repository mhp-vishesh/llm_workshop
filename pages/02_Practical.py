import huggingface_hub
from huggingface_hub import hf_hub_download
import os
import requests
import io
import asyncio
import edge_tts
import streamlit as st
import PyPDF2
import re
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TextStreamer
)
import base64
from huggingface_hub import login

# Monkey patch cached_download to replace removed function
def cached_download(url_or_filename, cache_dir=None, **kwargs):
    if str(url_or_filename).startswith(("http://", "https://")):
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        os.makedirs(cache_dir, exist_ok=True)
        filename = os.path.basename(url_or_filename)
        cache_path = os.path.join(cache_dir, filename)
        if os.path.exists(cache_path):
            return cache_path
        response = requests.get(url_or_filename, stream=True)
        response.raise_for_status()
        with open(cache_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return cache_path
    else:
        return hf_hub_download(repo_id=url_or_filename, **kwargs)

huggingface_hub.cached_download = cached_download



# Load embedding model
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Load quantized Llama-2 7B model with 4-bit quantization and set pad_token
@st.cache_resource(show_spinner=False)
def load_llama_model(model_name="meta-llama/Llama-2-7b-chat-hf"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token   # FIX for pad token error
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model

embedding_model = load_embedding_model()
tokenizer, llama_model = load_llama_model()

MAX_CONTEXT_TOKENS = 1024
MAX_CHUNK_SIZE = 1000

def extract_text_from_pdf(file) -> str:
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + " "
    return text

def clean_text(txt: str) -> str:
    txt = re.sub(r"-\s+", "", txt)
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()

def chunk_text(text, max_chunk_size=MAX_CHUNK_SIZE):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 2 > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
        else:
            current_chunk += sentence + '. '
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def embed_chunks(chunks):
    return embedding_model.encode(chunks, convert_to_tensor=True)

def build_context(chunks, tokenizer, max_tokens=MAX_CONTEXT_TOKENS):
    context = ""
    total_tokens = 0
    for chunk in chunks:
        chunk_tokens = len(tokenizer.tokenize(chunk))
        if total_tokens + chunk_tokens > max_tokens:
            break
        context += chunk + " "
        total_tokens += chunk_tokens
    return context.strip()

def llama_generate_answer(system_prompt, question_text, context_text, tokenizer, model, max_length=100, temperature=0.7):
    full_prompt = (
        f"{system_prompt}\n\nContext: {context_text}\n\nQuestion: {question_text}\nAnswer:"
    )
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            streamer=streamer,
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("Answer:")[-1].strip()

# Edge TTS functions for speech output
async def tts_to_bytesio(text, voice="en-US-AriaNeural"):
    import edge_tts
    mp3_buffer = io.BytesIO()
    communicate = edge_tts.Communicate(text, voice)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_buffer.write(chunk["data"])
    mp3_buffer.seek(0)
    return mp3_buffer

def sync_tts(text, voice="en-US-AriaNeural"):
    return asyncio.run(tts_to_bytesio(text, voice))

def play_audio_from_bytesio(mp3_buffer, key=None):
    if mp3_buffer is None:
        st.info("Audio playback unavailable due to TTS error.")
        return
    audio_bytes = mp3_buffer.read()
    b64_encoded = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
    <audio controls key="{key}">
        <source src="data:audio/mp3;base64,{b64_encoded}" type="audio/mp3" />
        Your browser does not support the audio element.
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)


# UI Start
st.set_page_config(page_title="LLaMA PDF Q&A with Speech", layout="wide")
st.title("📄 Hands-on Session")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if "uploaded_file_name" not in st.session_state:
    st.session_state["uploaded_file_name"] = None

if uploaded_file is not None:
    if st.session_state["uploaded_file_name"] != uploaded_file.name:
        with st.spinner("Extracting and embedding PDF text..."):
            text = extract_text_from_pdf(uploaded_file)
            text = clean_text(text)
            chunks = chunk_text(text)
            embeddings = embed_chunks(chunks)
            st.session_state["doc_chunks"] = chunks
            st.session_state["doc_embeddings"] = embeddings
            st.session_state["uploaded_file_name"] = uploaded_file.name
            st.session_state["last_answer"] = None
        st.success("✅ PDF text extracted and embedded.")

if st.session_state.get("doc_chunks") is not None and st.session_state.get("doc_embeddings") is not None:
    st.subheader("🧩 Extracted Text Chunks")
    with st.expander("View extracted chunks"):
        for i, chunk in enumerate(st.session_state["doc_chunks"]):
            st.write(f"Chunk {i+1}:")
            st.write(chunk)
            st.markdown("---")

    default_prompt = (
        "You are a knowledgeable and insightful virtual assistant with comprehensive expertise on Tacan cars. "
        "Always provide responses that remain contextually relevant to the information provided, "
        "maintaining a courteous and respectful tone throughout."
    )
    prompt_options = {
        "Courteous Expert (Default)": default_prompt,
        "Rude & Sarcastic Assistant": (
            "You are a very rude and sarcastic assistant. Always provide blunt, curt, and occasionally snarky responses."
        ),
        "Cheerful Assistant": (
            "You are a cheerful, upbeat AI assistant. Always answer with optimism, enthusiasm, and polite encouragement."
        ),
        "Dull & Unenthusiastic Assistant": (
            "You are uninterested, unenthusiastic, and respond in a monotone, lifeless way."
        ),
        "Disrespectful Assistant": (
            "You are a disrespectful assistant who is dismissive and ignores social niceties."
        ),
        "Custom...": None
    }
    with st.expander("🎭 Choose Assistant Personality / Prompt", expanded=False):
        persona = st.selectbox(
            "Select a persona for the assistant:",
            list(prompt_options.keys()),
            key="persona_select"
        )
        if persona == "Custom...":
            user_prompt = st.text_area(
                "Write your own assistant prompt:",
                value="You are a creative and helpful AI.",
                height=100,
                key="custom_prompt"
            )
        else:
            user_prompt = prompt_options[persona]
        st.markdown(f"**Active persona prompt:**\n\n> {user_prompt}")

    st.subheader("🎛️ Hyperparameters")
    col1, col2 = st.columns(2)
    temperature = col1.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    max_tokens = col2.slider("Max tokens", 50, 150, 100, 10)
    question = st.text_input("🔎 Ask a question about the PDF:")

    if st.button("🎤 Get Answer with Speech"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer with LLaMA..."):
                q_emb = embedding_model.encode(question, convert_to_tensor=True)
                hits = util.semantic_search(
                    q_emb, st.session_state["doc_embeddings"], top_k=8
                )[0]
                selected_chunks = [st.session_state["doc_chunks"][hit["corpus_id"]] for hit in hits]
                context_clean = build_context(selected_chunks, tokenizer, max_tokens=MAX_CONTEXT_TOKENS)

                answer = llama_generate_answer(
                    user_prompt, question, context_clean, tokenizer, llama_model,
                    max_length=max_tokens, temperature=temperature
                )
                st.session_state["last_answer"] = answer

    if st.session_state.get("last_answer"):
        st.markdown("### 💬 Text Answer")
        st.write(st.session_state["last_answer"])
        st.markdown("### 🎧 Speech Output")
        audio_buffer = sync_tts(st.session_state["last_answer"])
        play_audio_from_bytesio(audio_buffer, key=f"audio_{hash(st.session_state['last_answer'])}")
else:
    st.info("Please upload a PDF to begin.")
