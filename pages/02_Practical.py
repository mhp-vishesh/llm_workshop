import streamlit as st
import PyPDF2
import re
import io
from sentence_transformers import SentenceTransformer, util
from gtts import gTTS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import base64

# ----------------------------
# Model loading
# ----------------------------

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")




"""
@st.cache_resource(show_spinner=False)
def load_llama7b_model():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")  # or corresponding repo
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b",
        device_map="auto",
        load_in_8bit=True,              # if bitsandbytes quantization desired
        # or other quantization configs like 4-bit with BitsAndBytesConfig
    )
    return tokenizer, model




#original code what works


@st.cache_resource(show_spinner=False)
def load_distilgpt2_model():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "distilgpt2",
        device_map="auto"
    )
    model.eval()
    return tokenizer, model


embedding_model = load_embedding_model()
tokenizer, distilgpt2_model = load_distilgpt2_model()






"""













@st.cache_resource(show_spinner=False)
def load_distilgpt2_model():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b",
        device_map="auto",
        load_in_8bit=True, 
    )
    return tokenizer, model


embedding_model = load_embedding_model()
tokenizer, distilgpt2_model = load_distilgpt2_model()

# ----------------------------
# Helpers
# ----------------------------

MAX_CONTEXT_TOKENS = 1024
MAX_CHUNK_SIZE = 1000  # characters approx
RESERVED_TOKENS_FOR_OUTPUT = 128  # keep output room in context length

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

def build_context(chunks, tokenizer, max_tokens=MAX_CONTEXT_TOKENS, reserved_tokens=RESERVED_TOKENS_FOR_OUTPUT):
    context = ""
    total_tokens = 0
    max_available = max_tokens - reserved_tokens
    for chunk in chunks:
        chunk_tokens = len(tokenizer.encode(chunk))
        if total_tokens + chunk_tokens > max_available:
            break
        context += chunk + " "
        total_tokens += chunk_tokens
    return context.strip()


def distilgpt2_generate_answer(user_prompt, question, context, tokenizer, model, max_new_tokens=200, temperature=0.7):
    full_prompt = f"{user_prompt}\nQuestion: {question}\nContext: {context}\nAnswer:"
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True).to(model.device)
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=3,
            repetition_penalty=1.2
        )
    except Exception as e:
        return f"Generation error: {e}"
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



def tts_to_bytesio(text, lang="en", tld="com"):
    try:
        mp3_buffer = io.BytesIO()
        tts = gTTS(text=text, lang=lang, tld=tld)
        tts.write_to_fp(mp3_buffer)
        mp3_buffer.seek(0)
        return mp3_buffer
    except Exception as e:
        st.error(f"Text to speech error: {e}")
        return None

def play_audio_from_bytesio(mp3_buffer):
    if mp3_buffer is None:
        st.info("Audio playback unavailable due to TTS error.")
        return
    audio_bytes = mp3_buffer.read()
    b64_encoded = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
    <audio controls>
    <source src="data:audio/mp3;base64,{b64_encoded}" type="audio/mp3" />
    Your browser does not support the audio element.
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# --------------------------------------------------
# UI Start
# --------------------------------------------------

st.set_page_config(page_title="DistilGPT2 PDF Q&A with Speech", layout="wide")
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
    max_tokens = col2.slider("Max tokens", 50, 300, 256, 10)
    question = st.text_input("🔎 Ask a question about the PDF:")
    if st.button("🎤 Get Answer with Speech"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            if tokenizer is None or distilgpt2_model is None:
                st.error("Model is not loaded correctly. Please check.")
            else:
                with st.spinner("Generating answer..."):
                    q_emb = embedding_model.encode(question, convert_to_tensor=True)
                    hits = util.semantic_search(q_emb, st.session_state["doc_embeddings"], top_k=8)[0]
                    selected_chunks = [st.session_state["doc_chunks"][hit["corpus_id"]] for hit in hits]
                    context_clean = build_context(selected_chunks, tokenizer, max_tokens=MAX_CONTEXT_TOKENS)
                    answer = distilgpt2_generate_answer(
                                    user_prompt, question, context_clean, tokenizer, distilgpt2_model,
                                    max_new_tokens=int(max_tokens), temperature=temperature
                                )

                    st.session_state["last_answer"] = answer

    if st.session_state.get("last_answer"):
        st.markdown("### 💬 Text Answer")
        st.write(st.session_state["last_answer"])
        st.markdown("### 🎧 Speech Output")
        audio_buffer = tts_to_bytesio(st.session_state["last_answer"])
        play_audio_from_bytesio(audio_buffer)

else:
    st.info("Please upload a PDF to begin.")
