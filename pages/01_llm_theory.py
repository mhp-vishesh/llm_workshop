import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from transformers import GPT2TokenizerFast, GPT2Tokenizer, GPT2Model
import torch
import plotly.express as px
import time
import base64
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA

# ------------------------
# Cached model loading
# ------------------------
@st.cache_resource
def load_gpt2_model_and_tokenizers():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_attentions=True)
    tokenizer_fast = GPT2TokenizerFast.from_pretrained("gpt2")
    return tokenizer, model, tokenizer_fast

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_bert_model():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased', output_attentions=True)
    return tokenizer, model


# 1) Let's Learn Theory in Fun!
def display_pdf_in_expander(file_path):
    with st.expander("DataBricks_LLM_eBook", expanded=True):
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'''
            <iframe src="data:application/pdf;base64,{base64_pdf}" 
                style="width: 100%; height: 900px;" 
                frameborder="0" 
                scrolling="auto"
            >
            </iframe>
        '''
        st.markdown(pdf_display, unsafe_allow_html=True)

    with st.expander("Show PDF display code"):
        code = '''
def display_pdf_in_expander(file_path):
    with st.expander("DataBricks_LLM_eBook", expanded=True):
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f"""..."""
        st.markdown(pdf_display, unsafe_allow_html=True)
'''
        st.code(code, language='python')


# 2) LLM Overview
def llm_overview_section():
    with st.expander("LLM Overview", expanded=True):
        st.markdown("""
### What Are Large Language Models (LLMs)?
- Large Language Models (LLMs) process massive datasets to understand and generate human-like text.
- They use transformer architecture with attention mechanisms.
- Examples include GPT, BERT, and others.

### Foundation vs. Fine-Tuned Models
- Foundation models are pre-trained on large, diverse datasets.
- Fine-tuned models adapt foundation models for specific tasks or domains.
- Fine-tuning improves task-specific performance.

### Evolution of LLMs
- Early NLP models date back to the 1960s with pattern recognition approaches.
- Introduction of neural networks and LSTMs brought significant improvements.
- Transformers revolutionized the field since 2017.
- GPT series, BERT, and others progressed rapidly since then.

### Enterprise Benefits
- Automate support and customer interactions.
- Enhanced content creation and summarization.
- Data-driven decision making improvements.
- Scalability and efficiency in language understanding.

### Challenges with LLM
- Huge computational resources needed for training.
- Risks of bias and fairness issues.
- Difficulty in explainability and control.
- Data privacy and regulation compliance concerns.

### Evaluation Methods
- Standard Natural Language Understanding benchmarks.
- Human evaluation and feedback.
- Robustness and generalization testing.
- Real-world deployment monitoring.
""")


# 3) Tokenization
@st.cache_data
def create_tokenization_gif(_tokenizer_fast, sentence: str) -> str | None:
    tokens = [_t.replace("Ġ", "") for _t in _tokenizer_fast.tokenize(sentence)]
    if not tokens:
        return None

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#b8860b']
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.axis('off')

    char_width, spacing, x = 0.015, 0.02, 0.05
    x_positions = []
    for word in tokens:
        x_positions.append(x)
        x += len(word) * char_width + spacing

    text_elems = [ax.text(0, 0.5, "", ha="left", va="center", fontsize=16) for _ in tokens]

    def update(frame):
        for i, txt in enumerate(text_elems):
            if i <= frame:
                txt.set_text(tokens[i])
                txt.set_color(colors[i % len(colors)])
                txt.set_position((x_positions[i], 0.5))
            else:
                txt.set_text("")
        if frame >= len(tokens):
            ax.set_title(f"Total Tokens: {len(tokens)}", fontsize=14)
        return text_elems

    ani = FuncAnimation(fig, update, frames=len(tokens) + 3, interval=800, repeat=False)
    ani.save("tokenization.gif", writer=PillowWriter(fps=1))
    plt.close(fig)
    return "tokenization.gif"


def tokenization_section(tokenizer_fast):
    st.header("LLM Tokenization Visualization")
    sentence = st.text_input("Enter a sentence:", "The weather changes every day and often surprises us.")
    if sentence.strip():
        gif_path = create_tokenization_gif(tokenizer_fast, sentence)
        if gif_path:
            st.image(gif_path)
    else:
        st.warning("⚠️ Please enter a sentence.")


# 4) Chunking
def chunk_text(text):
    return [s.strip() for s in text.split(".") if s.strip()]

def chunking_section():
    st.header("Chunking")
    text = st.text_area("Enter text to chunk", "Attention is all you need. This sentence demonstrates chunking, embeddings, and attention visualization.")
    chunks = chunk_text(text)
    st.write(f"Total chunks: {len(chunks)}")
    col_chunks = st.columns(min(len(chunks), 6)) if len(chunks) > 0 else []
    for idx, chunk in enumerate(chunks):
        with col_chunks[idx % 6]:
            st.markdown(f"**Chunk {idx+1}:**")
            st.write(chunk)
    st.info("👉 Chunking breaks long text into manageable pieces for processing.")
    return chunks


# 5) Vectorization Embeddings
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def vectorization_section(chunks):
    embedding_model = load_embedding_model()
    st.header("Vectorization (Embeddings) & PCA Projection")
    if chunks:
        embeddings = embedding_model.encode(chunks)
        if len(chunks) > 1:
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(embeddings)
            fig = px.scatter(
                x=reduced[:, 0], y=reduced[:, 1],
                text=chunks,
                title="PCA Projection of Chunk Embeddings",
                labels={"x": "PCA 1", "y": "PCA 2"},
            )
            fig.update_traces(marker=dict(size=12, color="blue"))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            **Explanation:**
            - Close points indicate chunks with similar meaning.
            - Axes represent directions of maximum variance in embedding space.
            """)
        else:
            st.info("Need at least 2 chunks to show PCA.")
    else:
        st.info("Add text to see embeddings.")


# 6) Attention matrix
@st.cache_resource
def load_bert_model():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased', output_attentions=True)
    return tokenizer, model

def attention_mechanism_expander():
    st.header("Understanding Attention in LLM")
    # How to Read & Key Takeaways
    
    
    
    
    
    with st.expander("What is Attention in Large Language Models?"):
        st.markdown("""
        ### What Is Attention?
        Attention lets the model focus on the most relevant parts of input text to better understand context and relationships between words.
        It dynamically weighs each word's importance relative to others, allowing for better interpretation of meaning.

        ### How It Works
        For each word, we compute:
        - **Query (Q)**: what the model is looking for
        - **Key (K)**: references representing other words
        - **Value (V)**: the actual content of words

        The model calculates attention scores between query and keys, normalizes these via softmax to get weights, then computes a weighted sum of values.
        """)

        st.markdown("### The Formula: Scaled Dot-Product Attention")
        st.latex(r"""
        \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
        """)

        st.markdown(r"""
        Where:
        - \( Q \) = Query matrix
        - \( K \) = Key matrix
        - \( V \) = Value matrix
        - \( d_k \) = dimension of key vectors

        ### Intuition
        Attention is like a spotlight that focuses on different parts of a sentence depending on which word is being considered, helping capture context effectively.

        ### Why Attention Is Important
        - Enables capturing long-range dependencies across words.
        - Allows parallel processing of inputs (unlike sequential models).
        - Forms the core of Transformer models powering modern LLMs like GPT and BERT.

        For more detail, explore:
        - [AI21 Labs - Attention Mechanisms](https://www.ai21.com/knowledge/attention-mechanisms-language-models/)
        - [Sebastian Raschka Blog - Self-Attention From Scratch](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html)
        """)
    
    with st.expander("Show Attention Mechanism Section Code"):
        code = '''
def attention_mechanism_expander():
    with st.expander("What is Attention in Large Language Models?"):
        st.markdown("""...""")
        st.markdown("### The Formula: Scaled Dot-Product Attention")
        st.latex(r"""...""")
        st.markdown(r"""...""")
'''
        st.code(code, language='python')

def attention_heatmap_section(tokenizer, model, chunks):
    st.header("Attention Heatmap Visualization")
    
    with st.expander("ℹ️ How to interpret attention & takeaways"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### How to Read an Attention Map")
            st.markdown("""
    - **Rows (Y-axis = Query tokens):** Each token asking: "Where should I look?"
    - **Columns (X-axis = Key tokens):** Tokens being looked at.
    - **Color intensity:** Strength of focus (yellow = strong, purple = weak).
    - **Diagonal dominance:** Tokens often attend strongly to themselves.
    - **[CLS] connections:** `[CLS]` is attended to often as it aggregates sentence meaning.
    - **[SEP] connections:** `[SEP]` usually has little meaning but marks sentence boundaries.
    - **Layers:** Lower layers capture word identity/grammar; higher layers capture abstract meaning.
    - **Heads:** Each head looks at different relationships (e.g., syntax, long-range dependencies).
            """)
        with col2:
            st.markdown("#### Key Takeaways")
            st.markdown("""
    1. Transformers don’t read sequentially — they *focus attention* on relevant words.
    2. Different heads learn different perspectives of relationships.
    3. Self-attention allows long-range dependencies (e.g., subject ↔ verb).
    4. PCA on embeddings shows semantic similarity (close = similar meaning).
    5. Attention maps explain *why* models make certain predictions.
            """)
    
    
    
    
    if chunks:
        attention_text = chunks[0]
        inputs = tokenizer(attention_text, return_tensors="pt")
        outputs = model(**inputs)
        attentions = outputs.attentions
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        st.markdown("**Select Layer and Head:**")
        selected_layer = st.slider("Layer", 0, len(attentions) - 1, 0)
        selected_head = st.slider("Head", 0, attentions[selected_layer].shape[1] - 1, 0)
        attn_matrix = attentions[selected_layer][0, selected_head].detach().numpy()

        hover_text = []
        for i, query in enumerate(tokens):
            row = []
            for j, key in enumerate(tokens):
                row.append(f"Query: {query}<br>Key: {key}<br>Weight: {attn_matrix[i,j]:.3f}")
            hover_text.append(row)

        fig = go.Figure(data=go.Heatmap(
            z=attn_matrix,
            x=tokens,
            y=tokens,
            colorscale='Viridis',
            hoverinfo="text",
            text=hover_text
        ))
        fig.update_layout(
            title=f"Attention Matrix (Layer {selected_layer}, Head {selected_head})",
            xaxis_title="Key Tokens",
            yaxis_title="Query Tokens",
            width=700,
            height=600
        )
        st.plotly_chart(fig)

        st.markdown("#### Layer and Head Details")
        st.info(
            f"""
**Layer {selected_layer}:**  
Lower layers focus more on local syntax or word-identity, while higher layers pay more attention to high-level meaning or long-distance relations.

**Head {selected_head}:**  
Each head learns a different way of connecting tokens—it might track subjects, verbs, punctuation, or other relationships.
            """
        )

        with st.expander("What are [CLS] and [SEP] tokens?"):
            st.markdown("""
**[CLS]:** Classification token added at input start.  
- Used for aggregating global sentence meaning (e.g., for classification tasks).

**[SEP]:** Separator token.  
- Used as a boundary between sentences or at the end.

You often see strong attention to [CLS] (for summarization) and [SEP] (for division between sentence parts). They are *special* tokens introduced during tokenization.
            """)
    else:
        st.info("Add text to see attention matrix.")
        
        
        
# ------------------------
# LLM Parameters Explanation
# ------------------------
def llm_parameters_explanation():
    with st.expander("Key LLM Parameters & Concepts Explained 📚🔍", expanded=False):
        st.markdown("""
        ### Temperature
        Controls randomness in the model's output generation.  
        - Low (e.g., 0.2): More focused and deterministic answers.  
        - High (e.g., 0.8): More creative and diverse outputs.

        ### Top-K Sampling
        Restricts token choices to the top K probable tokens at each generation step, improving output quality by excluding unlikely tokens.

        ### Top-P (Nucleus) Sampling
        Selects tokens from the smallest set accumulating probability P (e.g., 0.9), balancing creativity with coherence.

        ### Max Tokens
        The maximum number of tokens the model generates per request. Controls length and resource usage.

        ### Batch Size
        Number of samples processed simultaneously during training or inference. Larger batches speed up processing but require more memory.

        ### Epochs
        The number of full passes over the training dataset. More epochs typically improve learning but risk overfitting.

        ### Learning Rate
        The steps the optimizer takes during training to minimize loss. 
        - Too high: Training can be unstable.  
        - Too low: Training is slow and may get stuck.

        """)

        st.image("./resources/Tempearture.png", caption="Effect of Temperature on model creativity")
        st.image("./resources/Top_parameters.png", caption="Difference between Top-K and Top-P Sampling")
        st.markdown("""
        For more detailed explanations and interactive visualizations, refer to the external resources within this app and beyond.
        """)

    with st.expander("Show LLM Parameters Explanation code"):
        code = '''def llm_parameters_explanation():
    with st.expander("Key LLM Parameters & Concepts Explained 📚🔍", expanded=False):
        st.markdown(\"\"\"...\"\"\")
        # images and markdown...
'''
        st.code(code, language='python')

# 7) Chocolate animation
def plot_chocolate_animation(total_chocolates, batch_size, epochs, learning_rate):
    plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 14})
    steps_per_epoch = (total_chocolates + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * epochs
    plot_placeholder = st.empty()
    progress_bar = st.progress(0)
    fig_width, fig_height = 14, 4

    for step_num in range(total_steps):
        epoch, step_in_epoch = divmod(step_num, steps_per_epoch)
        chocolates_picked = min(batch_size, total_chocolates - step_in_epoch * batch_size)
        pos_current = step_num * learning_rate

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('off')

        ax.hlines(0, 0, total_steps * learning_rate, colors='#cccccc', linestyles='dashed', linewidth=2)
        ax.plot(0, 0, "o", markersize=16, color="#27ae60", label="Start", zorder=5)
        ax.plot(total_steps * learning_rate, 0, "o", markersize=16, color="#c0392b", label="End", zorder=5)

        for past_step in range(step_num + 1):
            p_epoch, p_step = divmod(past_step, steps_per_epoch)
            chocs_left = total_chocolates - p_step * batch_size
            chocs = min(batch_size, chocs_left)
            pos = past_step * learning_rate
            y_offset = -p_epoch * 0.8

            rect = patches.FancyBboxPatch(
                (pos, y_offset), learning_rate * 0.9, 0.5,
                boxstyle="round,pad=0.02",
                linewidth=1,
                facecolor='#6c5ce7',
                edgecolor='black',
                alpha=0.8,
                zorder=3,
            )
            ax.add_patch(rect)
            ax.text(pos + learning_rate * 0.45, y_offset + 0.25, f"{chocs}",
                    ha='center', va='center', fontsize=14, color='white')

        rect_cur = patches.FancyBboxPatch(
            (pos_current, 1.0), learning_rate * 0.9, 0.5,
            boxstyle="round,pad=0.05",
            linewidth=2,
            facecolor='#55efc4',
            edgecolor='black',
            zorder=6,
            alpha=0.95,
        )
        ax.add_patch(rect_cur)
        ax.text(pos_current + learning_rate * 0.45, 1.7, f"Active\n({chocolates_picked})",
                ha='center', va='bottom', fontsize=16, fontweight='bold', color='black', zorder=7)

        for e in range(epochs):
            ax.text(total_steps * learning_rate + learning_rate * 0.7, -e * 0.8 + 0.25,
                    f"Epoch {e + 1}", ha='left', fontsize=14, color='#636e72')

        ax.set_xlim(-learning_rate * 0.5, total_steps * learning_rate + learning_rate * 3)
        ax.set_ylim(-epochs * 0.9, 2)

        ax.legend(loc='lower left', bbox_to_anchor=(-0.08, -0.3), fontsize=14, frameon=True)

        plot_placeholder.pyplot(fig)
        progress_bar.progress((step_num + 1) / total_steps)
        time.sleep(0.2)

    st.success("🏁 Finished carrying all chocolates!")

def chocolate_animation_section():
    with st.container():
        with st.expander("Dynamic Visualization: Epoch, Batch Size & Learning Rate (Chocolate Analogy)", expanded=False):
            total_chocs = st.number_input("🍫 Total Chocolates (Dataset size)", min_value=1, value=32)
            batch = st.number_input("✋ Batch Size (Handful size)", min_value=1, max_value=total_chocs, value=4)
            ep = st.number_input("🔁 Epochs (Number of full passes)", min_value=1, value=5)
            lr = st.slider("📈 Learning Rate (Step size)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)

            col1, col2 = st.columns([3, 1])
            with col1:
                plot_chocolate_animation(total_chocs, batch, ep, lr)
            with col2:
                st.video("./resources/neural_network_working.mp4", format="video/mp4", start_time=0)

    with st.expander("Show Chocolate Animation Section code"):
        code = '''def chocolate_animation_section():
    with st.container():
        with st.expander("Dynamic Visualization: Epoch, Batch Size & Learning Rate (Chocolate Analogy)", expanded=False):
            total_chocs = st.number_input("🍫 Total Chocolates (Dataset size)", min_value=1, value=32)
            batch = st.number_input("✋ Batch Size (Handful size)", min_value=1, max_value=total_chocs, value=4)
            ep = st.number_input("🔁 Epochs (Number of full passes)", min_value=1, value=5)
            lr = st.slider("📈 Learning Rate (Step size)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)

            col1, col2 = st.columns([3, 1])
            with col1:
                plot_chocolate_animation(total_chocs, batch, ep, lr)
            with col2:
                st.video("./resources/neural_network_working.mp4", format="video/mp4", start_time=0)'''
        st.code(code, language='python')


# 8) Workshop resources dropdown
def workshop_resources_dropdown():
    with st.expander("Workshop Links & Resources", expanded=False):
        options = ["Select a resource", "Transformer Explainer (Interactive Demo)"]
        choice = st.selectbox("Workshop Links & Resources", options, index=0, key="workshop_resource_select")

        if choice == "Transformer Explainer (Interactive Demo)":
            st.markdown(
                '<iframe src="https://poloclub.github.io/transformer-explainer/" width="100%" height="600" frameborder="0" allowfullscreen></iframe>',
                unsafe_allow_html=True
            )

    with st.expander("Show Workshop Resources Dropdown code"):
        code = '''def workshop_resources_dropdown():
    with st.expander("Workshop Links & Resources", expanded=False):
        options = ["Select a resource", "Transformer Explainer (Interactive Demo)"]
        choice = st.selectbox("Workshop Links & Resources", options, index=0, key="workshop_resource_select")
    
        if choice == "Transformer Explainer (Interactive Demo)":
            st.markdown(
                '<iframe src="https://poloclub.github.io/transformer-explainer/" width="100%" height="600" frameborder="0" allowfullscreen></iframe>',
                unsafe_allow_html=True
            )'''
        st.code(code, language='python')


# ------------------------
# Main orchestrator
# ------------------------
def main():
    st.markdown("<h1 style='text-align: center;'>Let's learn theory in a fun way!!!!</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # 1. Teach theory with data ebook
    display_pdf_in_expander("./resources/LLM_guide.pdf")
    st.markdown("---")

    # 2. LLM overview section
    llm_overview_section()
    st.markdown("---")

    # 3. Load tokenizers/models for tokenization visualization
    tokenizer, model, tokenizer_fast = load_gpt2_model_and_tokenizers()

    # 4. Tokenization visualization with GIF
    tokenization_section(tokenizer_fast)
    st.markdown("---")

    # 5. Chunking
    chunks = chunking_section()
    st.markdown("---")

    # 6. Vectorization embeddings & PCA plot
    vectorization_section(chunks)
    st.markdown("---")

    # 7. Load tokenizer and model for attention
    tokenizer_bert, model_bert = load_bert_model()

    # 8. Attention mechanism explanation
    attention_mechanism_expander()
    st.markdown("---")

    # 9. Attention heatmap visualization
    attention_heatmap_section(tokenizer_bert, model_bert, chunks)
    st.markdown("---")

    # 10. LLM Parameters explanation
    llm_parameters_explanation()
    st.markdown("---")

    # 11. Chocolate animation analogy
    chocolate_animation_section()
    st.markdown("---")

    # 12. Workshop resources dropdown
    workshop_resources_dropdown()
    st.markdown("---")


if __name__ == "__main__":
    main()
