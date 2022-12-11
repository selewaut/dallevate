import requests
import streamlit as st
import openai
import os
import numpy as np
import streamlit.components.v1 as components
from streamlit_chat import message
from utils import (
    generate_query_variations_gpt,
    process_dalle_images,
    query_multiple_variations_dalle,
)

openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI()
base_image_dir = os.path.join("images", "01_generations")
original_prompt = "two dogs playing chess, oil painting"
n_variations = 1

openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI()

filepaths = [
    "images/02_template_images/generation_1.png",
    "images/02_template_images/generation_2.png",
    "images/02_template_images/generation_3.png",
    "images/02_template_images/generation_4.png",
]

st.set_page_config(
    page_title="OpenAI Streamlit Gallery",
    page_icon=":rocket:",
    layout="wide",
)


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


#  To get rid of the Streamlit branding stuff
local_css("css/styles.css")


#  Anchor
st.title(
    "#"
)  # This anchor is needed for the page to start at the top when it is called.

# --- INTRO ---

with st.container(height=150):

    st.title("Dalle Competition")
    st.subheader(
        """
        Insert prompt, generate alternative prompts and generate 4 DALLE images
        """
    )
    st.write("""""")

with st.container():
    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        st.header("Create AI Artwork")
        st.caption(
            "Note: Currently static image as OpenAI API query limit has been reached"
        )
        ai_image_idea = st.text_input(
            "Enter an Image to Create",
            "logo of an oil company",
            key="placeholder",
        )
        button = st.button("Generate Art 🏃", key="generate")
        if button and ai_image_idea:
            st.write("Generating Artwork...")
            # generate queries
            response_dict_gpt = generate_query_variations_gpt(
                original_prompt=ai_image_idea, n_variations=n_variations
            )

            image_responses = query_multiple_variations_dalle(response_dict_gpt)

            # process images
            filepaths = process_dalle_images(
                image_list=image_responses,
                filename="generation",
                image_dir=base_image_dir,
            )

    with col2:
        with st.spinner("Generating your Art"):
            print(len(filepaths), "LEN")
            idx = 0
            for _ in range(len(filepaths) - 1):
                cols = st.columns(2)

                if idx < len(filepaths):
                    cols[0].image(
                        filepaths[idx],
                        width=250,
                    )
                idx += 1

                if idx < len(filepaths):
                    cols[1].image(
                        filepaths[idx],
                        width=250,
                    )
                    idx = idx + 1
                else:
                    break
