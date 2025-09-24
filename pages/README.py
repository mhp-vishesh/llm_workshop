import streamlit as st

def readme_page():
    st.set_page_config(page_title="Project README", layout="wide")
    st.title("Large Language Models (LLM) Workshop App")

    readme_md = """
    Welcome to the LLM Workshop Streamlit app, a comprehensive interactive platform designed to teach and demonstrate Large Language Model concepts, theory, and practical usage through an intuitive web interface.

    ---

    ## Table of Contents

    - [Project Overview](#project-overview)
    - [Features](#features)
    - [Project Structure](#project-structure)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Dependencies](#dependencies)
    - [Contributing](#contributing)
    - [Contact](#contact)
    - [License](#license)

    ---

    ## Project Overview

    This app is a multipage Streamlit application that guides users through understanding and working with Large Language Models (LLMs). It features:

    - A **Main Landing Page** introducing the workshop themes.
    - An in-depth **Theory Page** covering LLM architectures, tokenization, attention mechanisms, and visualization tools.
    - A **Practical Page** for hands-on interaction with LLaMA-based PDF Q&A.
    - A **Helper Page** providing resource links and eBook uploads for extended learning.

    The app is ideal for learners and educators aiming to combine theoretical knowledge with practical demonstrations in an accessible GUI.

    ---

    ## Features

    - Interactive visualization of tokenization and attention mechanisms.
    - Hands-on PDF Q&A powered by a local LLaMA LLM.
    - Upload and categorize eBooks with curated LLM-related external learning materials.
    - Responsive UI with headings, expanders, and multimedia content.
    - Modular, maintainable codebase split into clearly defined pages.

    ---

    ## Project Structure

    ```
    .
    ├── AGENDA.py           # Main landing page
    ├── pages
        ├── 01_LLM_THEORY.py   # LLM Theory and visualization page
        ├── 02_PRACTICAL.py    # Practical PDF Q&A page using LLaMA
        ├── 03_HELPER.py       # Helper page with eBooks and links
        ├── README.md         # This documentation
    ├── requirements.txt  # Python dependencies
    └── resources/        # Images and static media assets
    ```

    ---

    ## Installation

    1. Clone this repository:

        ```
        git clone https://github.com/mhp-vishesh/llm_workshop
        cd llm_workshop
        ```

    2. Create and activate a Python virtual environment:

        ```
        python -m venv venv
        source venv/bin/activate  # macOS/Linux
        # or
        venv\\Scripts\\activate  # Windows
        ```

    3. Install dependencies:

        ```
        pip install -r requirements.txt
        ```

    ---

    ## Usage

    Run the Streamlit app locally:

    ```
    streamlit run main.py
    ```

    Use the sidebar navigation to switch between:

    - **AGENDA:** Overview and workshop introduction.
    - **LLM Theory:** Dive deep into LLM concepts and visualizations.
    - **PRACTICAL:** Interact with LLaMA PDF Q&A.
    - **HELPER:** Upload eBooks and explore curated LLM resources.

    ---

    ## Dependencies

    Key Python packages used in this project include:

    - `streamlit` - Web app framework.
    - `transformers` - Hugging Face Transformers library with GPT2 and LLaMA models.
    - `sentence-transformers` - Embedding models for semantic search.
    - `gtts` - Google Text-to-Speech for speech output.
    - `matplotlib`, `plotly` - Visualization libraries.
    - `torch` - PyTorch backend for model inference.
    - `PyPDF2` - PDF text extraction utility.

    ---

    ## Contact

    Project Maintainer - [Vishesh Breja]  
    Email: vishesh.breja@mhp.com 

    ---

    Thank you for exploring the LLM Workshop app! We hope it provides a valuable learning experience combining theory and practice.
    """

    # Render the above markdown content
    st.markdown(readme_md, unsafe_allow_html=True)

if __name__ == "__main__":
    readme_page()
