

import streamlit as st


# PAGE CONFIGURATION
st.set_page_config(
    page_title="AstraMed",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SIDEBAR
with st.sidebar:
    st.markdown(
        """
        <h1 style='text-align:center;'>
        🫁 AstraMed
        </h1>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### Version")
    st.success("v1.1")
    st.markdown("### Dataset")
    st.info("RSNA Pneumonia Detection Challenge")
    st.markdown("### Region")
    st.info("Thorax")
    st.markdown("### Disease")
    st.info("Pneumonia")
    st.markdown("---")
    st.markdown("### Sprint")
    st.success("Sprint 3.5")
    st.markdown("---")
    st.caption("Built Using Streamlit")

# MAIN PAGE
st.title("🫁 AstraMed")

st.subheader(    "AI-powered Medical Imaging Platform for Pneumonia Detection")

st.write(
    """
    AstraMed is an end-to-end Medical AI project focused on
    Pneumonia Detection using Chest X-rays.

    This dashboard allows interactive exploration of the
    RSNA Pneumonia Detection Challenge dataset.
    """
)

st.markdown("---")

# DASHBOARD CARDS
col1 , col2 , col3 = st.columns(3)

with col1:
    st.metric(
        label = 'Dataset',
        value = 'RSNA'
    )

with col2:
    st.metric(
        label = 'Task',
        value="Pneumonia Classification",
    )

with col3:
    st.metric(
        label = "Status",
        value = "EDA Complete ✅ "
    )

st.markdown("---")


# PROJECT ROADMAP
st.header("📌 Project Roadmap")

st.markdown("""

✅ Sprint 1 — Dataset Understanding

✅ Sprint 2 — Data Engineering

✅ Sprint 3 — Exploratory Data Analysis

🔄 Sprint 3.5 — Dashboard

⬜ Sprint 4 — Modeling

⬜ Sprint 5 — Training

⬜ Sprint 6 — Evaluation

⬜ Sprint 7 — Explainability

⬜ Sprint 8 — Deployment

""")


# Dashboard Features
st.header("🚀 Dashboard Features")

features = [

    "📊 Dataset Statistics",

    "🩻 Image Explorer",

    "📁 Metadata Explorer",

    "📦 Bounding Box Viewer",

    "🔍 Patient Search"

]

for feature in features:

    st.write(feature)


# FOOTER
st.markdown("---")

st.caption(
    """
        AstraMed v1.1

        Built using Python • PyTorch • Streamlit • RSNA Dataset
    """
)


# 