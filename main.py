import streamlit as st
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
#from matplotlib.animation import FuncAnimation, PillowWriter
#import numpy as np
#from transformers import GPT2TokenizerFast, GPT2Tokenizer, GPT2Model
#import torch
#import plotly.express as px
#import time




###
# Testing
###


from databricks.sdk import WorkspaceClient

def test_model_call():
    try:
        client = WorkspaceClient()
        response = client.foundation_model.completions.create(
            model="databricks-meta-llama-3-1-8b-instruct",
            prompt="Hello, test",
            max_tokens=10,
        )
        return response.choices[0].text
    except Exception as e:
        return f"Error: {e}"

def main():
    st.set_page_config(page_title="LLM Workshop", page_icon="🚀", layout="wide")
    st.title("LLM Workshop with Databricks")

    # Your existing UI functions
    # render_header()
    # render_overview_intro()
    # render_overview_expanders()

    output = test_model_call()
    st.markdown("### Model test output:")
    st.write(output)




















# ------------------------
# UI Components
# ------------------------
def render_header():
    col1, col2, col3 = st.columns([3, 1, 3], vertical_alignment="center")
    with col1:
        st.image("./resources/LLM.jpeg")
    with col2:
        st.markdown("<h1 style='text-align:center; color:white;'>@</h1>", unsafe_allow_html=True)
    with col3:
        st.image("./resources/MHP_LOGO.png")
    st.markdown(" ")  # Spacer

def render_overview_intro():
    st.markdown("<h1 style='text-align: center;'>Welcome to the LLM Workshop</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.video("https://www.youtube.com/watch?v=LPZh9BOjkQs")
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; font-size: 24px; font-weight: bold;'>"
        "This workshop is designed to address the following themes"
        "</div>",
        unsafe_allow_html=True,
    )

def render_overview_expanders():
    c1, c2, c3 = st.columns(3)
    with c1:
        with c1:
            with st.expander("LLM IN THEORY"):
                st.markdown("""
                    - What Are Large Language Models (LLMs)?
                        - Foundation vs. Fine-Tuned Models
                        - Evolution of LLMs
                        - Enterprise Benefits
                        - Challenges
                        - Evaluation Methods
                    - LLM Tokenization
                    - Attention in LLM's
                    - Key LLM Parameters
                    - Training Parameters
                    """)
    with c2:
        with st.expander("LLM PRACTICAL SESSION"):
            st.write("Hands-on session")
    with c3:
        with st.expander("HELPER"):
            st.write("Helpful links or E-books")

# ------------------------
# Main function
# ------------------------
def main():
    st.set_page_config(page_title="LLM Workshop", page_icon="🚀", layout="wide")
    render_header()
    render_overview_intro()
    render_overview_expanders()

if __name__ == "__main__":
    main()
