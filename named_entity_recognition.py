import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Multilingual NER (WikiNEuRal)",
    page_icon="🔎",
    layout="centered"
)

MODEL_NAME = "Babelscape/wikineural-multilingual-ner"

ENTITY_COLORS = {
    "PER": "#ffadad",   # Person - pink
    "LOC": "#a0e7e5",   # Location - turquoise
    "ORG": "#ffd6a5",   # Organization - light orange
    "MISC": "#caffbf"   # Miscellaneous - light green
}

EXAMPLES = [
    "Elon Musk is the CEO of Tesla and SpaceX, and he was born in South Africa.",
    "Emmanuel Macron est le président de la France et il a visité Paris hier.",
    "Lionel Messi juega para el Inter Miami y nació en Rosario, Argentina."
]


# -------------------------------------------------
# Load model (cached so it only loads once per session)
# -------------------------------------------------
@st.cache_resource(show_spinner="Loading model, please wait...")
def load_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    ner = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )
    return ner


ner_pipeline = load_pipeline()


# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def extract_entities(text):
    """Runs the NER pipeline and returns a DataFrame of entities."""
    results = ner_pipeline(text)
    if not results:
        return pd.DataFrame(columns=["Entity", "Type", "Confidence"])

    return pd.DataFrame([
        {
            "Entity": r["word"],
            "Type": r["entity_group"],
            "Confidence": round(float(r["score"]), 4)
        }
        for r in results
    ])


def build_highlighted_html(text):
    """Returns the text as HTML with entities highlighted by type."""
    if not text or not text.strip():
        return "<p>Please enter some text first.</p>"

    results = ner_pipeline(text)
    results = sorted(results, key=lambda r: r["start"])

    if not results:
        return f'<div style="font-size:16px;">{text}</div><p><i>No entities were found.</i></p>'

    html_text = ""
    last_idx = 0

    for r in results:
        start, end = r["start"], r["end"]
        entity_type = r["entity_group"]
        color = ENTITY_COLORS.get(entity_type, "#e0e0e0")

        html_text += text[last_idx:start]
        html_text += (
            f'<span style="background-color:{color}; padding:2px 4px; '
            f'border-radius:4px; margin:0 1px;">'
            f'{text[start:end]} <b style="font-size:10px;">[{entity_type}]</b>'
            f'</span>'
        )
        last_idx = end

    html_text += text[last_idx:]
    return f'<div style="font-size:17px; line-height:2;">{html_text}</div>'


# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("🔎 Multilingual Named Entity Recognition")
st.markdown(
    "This app uses the **`Babelscape/wikineural-multilingual-ner`** model to detect "
    "named entities (Persons, Locations, Organizations, and Miscellaneous) in text. "
    "It supports **9 languages**: German, English, Spanish, French, Italian, Dutch, "
    "Polish, Portuguese, and Russian."
)

st.markdown("**Entity legend:** "
            "🟥 PER (Person) &nbsp;&nbsp; 🟦 LOC (Location) &nbsp;&nbsp; "
            "🟧 ORG (Organization) &nbsp;&nbsp; 🟩 MISC (Miscellaneous)")

st.divider()

# Example buttons
st.write("**Try an example:**")
cols = st.columns(len(EXAMPLES))
if "user_text" not in st.session_state:
    st.session_state.user_text = ""

for i, example in enumerate(EXAMPLES):
    if cols[i].button(f"Example {i + 1}", use_container_width=True):
        st.session_state.user_text = example

user_text = st.text_area(
    "Enter your text:",
    value=st.session_state.user_text,
    height=120,
    placeholder="Type a sentence in any of the 9 supported languages..."
)

analyze_clicked = st.button("Analyze", type="primary")

if analyze_clicked or user_text:
    if user_text.strip():
        with st.spinner("Analyzing text..."):
            html_output = build_highlighted_html(user_text)
            df_output = extract_entities(user_text)

        st.subheader("Highlighted Text")
        st.markdown(html_output, unsafe_allow_html=True)

        st.subheader("Extracted Entities")
        if df_output.empty:
            st.info("No entities were found in this text.")
        else:
            st.dataframe(df_output, use_container_width=True, hide_index=True)
    else:
        st.warning("Please enter some text to analyze.")

st.divider()
st.caption(
    "Model: Babelscape/wikineural-multilingual-ner — "
    "Tedeschi et al. (2021), EMNLP Findings 2021."
)
