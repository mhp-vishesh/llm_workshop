import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
#import numpy as np
from transformers import GPT2TokenizerFast, GPT2Tokenizer, GPT2Model
import torch
import plotly.express as px
import time
import base64






# ------------------------
# Cached model loading
# ------------------------
@st.cache_resource
def load_gpt2_model_and_tokenizers():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_attentions=True)
    tokenizer_fast = GPT2TokenizerFast.from_pretrained("gpt2")
    return tokenizer, model, tokenizer_fast


#--------------------------
#LLM Theory Text
#--------------------------


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

# Usage:











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










# ------------------------
# Cached tokenization GIF creator
# ------------------------
@st.cache_data
def create_tokenization_gif(_tokenizer_fast, sentence: str) -> str | None:
    # Tokenize and clean tokens for display
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

# ------------------------
# Attention mechanism explanation
# ------------------------
def attention_mechanism_expander():
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

# ------------------------
# Attention heatmap with Plotly
# ------------------------
def plot_attention_heatmap(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    attn = outputs.attentions[0][0, 0].numpy()

    fig = px.imshow(
        attn,
        labels=dict(x="Key Token", y="Query Token", color="Attention"),
        x=[tokenizer.decode([i]) for i in inputs.input_ids[0]],
        y=[tokenizer.decode([i]) for i in inputs.input_ids[0]],
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)

def attention_heatmap_section(tokenizer, model):
    st.header("Attention Heatmap Visualization")
    text = st.text_area("Enter text", "Deep learning transforms AI.")
    if text.strip():
        plot_attention_heatmap(text, tokenizer, model)
        
        
        
        
        
# ------------------------
# LLM Parameters
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

        # Suggested visualizations or images URLs (replace with local if you have)
        st.markdown("### Visualizing Concepts")
        
        # Visual: Temperature effect (conceptual image)
        st.image(
            "./resources/Tempearture.png",
            caption="Effect of Temperature on model creativity",
          
        )
        
        st.image(
            "./resources/Top_parameters.png",
            caption="Difference between Top-K and Top-P Sampling",
          
        )
        st.markdown("""
        For more detailed explanations and interactive visualizations, refer to the external resources within this app and beyond.
        """)

# Usage: Call `llm_parameters_explanation()` inside your chocolate_animation_section before rendering controls




# ------------------------
# Chocolate animation analogy
# ------------------------
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

        # Progress line
        ax.hlines(0, 0, total_steps * learning_rate, colors='#cccccc', linestyles='dashed', linewidth=2)
        ax.plot(0, 0, "o", markersize=16, color="#27ae60", label="Start", zorder=5)
        ax.plot(total_steps * learning_rate, 0, "o", markersize=16, color="#c0392b", label="End", zorder=5)

        # Chocolates as rounded rectangles
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

        # Highlight current step
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

        # Epoch labels on right
        for e in range(epochs):
            ax.text(total_steps * learning_rate + learning_rate * 0.7, -e * 0.8 + 0.25,
                    f"Epoch {e + 1}", ha='left', fontsize=14, color='#636e72')

        # Set limits
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

            # Create two columns side-by-side
            col1, col2 = st.columns([3, 1])  # adjust widths as needed
            
            # Left column: chocolate animation plot and progress
            with col1:
                plot_chocolate_animation(total_chocs, batch, ep, lr)
            
            # Right column: video player
            with col2:
                st.video("./resources/neural_network_working.mp4", format="video/mp4", start_time=0)






# ------------------------
# Workshop resources dropdown
# ------------------------
def workshop_resources_dropdown():
    with st.expander("Workshop Links & Resources", expanded=False):
        options = ["Select a resource", "Transformer Explainer (Interactive Demo)"]
        choice = st.selectbox("Workshop Links & Resources", options, index=0, key="workshop_resource_select")
    
        if choice == "Transformer Explainer (Interactive Demo)":
            st.markdown(
                '<iframe src="https://poloclub.github.io/transformer-explainer/" width="100%" height="600" frameborder="0" allowfullscreen></iframe>',
                unsafe_allow_html=True
            )

# ------------------------
# Main orchestrator
# ------------------------
def main():
    
    st.markdown("<h1 style='text-align: center;'>Let us learn theory in a fun way!!!!</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    
    
    display_pdf_in_expander("./resources/LLM_guide.pdf")
    
    st.markdown("---")
    
    llm_overview_section()
    tokenizer, model, tokenizer_fast = load_gpt2_model_and_tokenizers()
    tokenization_section(tokenizer_fast)
    attention_heatmap_section(tokenizer, model)
    attention_mechanism_expander()
    llm_parameters_explanation()
    chocolate_animation_section()
    workshop_resources_dropdown()

if __name__ == "__main__":
    main()
