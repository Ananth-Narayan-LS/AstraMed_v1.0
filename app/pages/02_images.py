"""
02_Image_Explorer.py

Interactive DICOM Image Explorer
"""

import random

import streamlit as st

from src.data.labels import load_labels
from src.data.dicom_loader import load_dicom
from src.utils.config import TRAIN_IMAGES, TRAIN_LABELS


# ==========================================================
# CACHE
# ==========================================================

@st.cache_data
def load_dataset():
    labels, _ = load_labels(TRAIN_LABELS)
    patient_ids = sorted(labels.keys())
    return labels, patient_ids


labels, patient_ids = load_dataset()


# ==========================================================
# PAGE TITLE
# ==========================================================

st.title("🩻 Image Explorer")

st.markdown(
    """
Browse chest X-rays from the **RSNA Pneumonia Detection Challenge** dataset.
"""
)

st.divider()


# ==========================================================
# SESSION STATE
# ==========================================================

if "selected_index" not in st.session_state:
    st.session_state.selected_index = 0


# ==========================================================
# NAVIGATION
# ==========================================================

col_prev, col_random, col_next = st.columns(3)

with col_prev:
    if st.button("⬅ Previous"):
        if st.session_state.selected_index > 0:
            st.session_state.selected_index -= 1

with col_random:
    if st.button("🎲 Random"):
        st.session_state.selected_index = random.randint(
            0,
            len(patient_ids) - 1
        )

with col_next:
    if st.button("Next ➡"):
        if st.session_state.selected_index < len(patient_ids) - 1:
            st.session_state.selected_index += 1


st.caption(
    f"Image {st.session_state.selected_index + 1:,} of {len(patient_ids):,}"
)


# ==========================================================
# SLIDER
# ==========================================================

selected_index = st.slider(
    "Select Image",
    min_value=0,
    max_value=len(patient_ids) - 1,
    value=st.session_state.selected_index,
)

st.session_state.selected_index = selected_index

patient_id = patient_ids[selected_index]


# ==========================================================
# LOAD IMAGE
# ==========================================================

image_path = TRAIN_IMAGES / f"{patient_id}.dcm"

image, metadata = load_dicom(image_path)


# ==========================================================
# DISPLAY
# ==========================================================

col1, col2 = st.columns([2.5, 1])

with col1:

    st.image(
        image,
        caption=f"Patient ID : {patient_id}",
        width="stretch",
        clamp=True,
    )

with col2:

    st.subheader("Patient Information")

    st.write(f"**Patient ID:** `{patient_id}`")

    target = labels[patient_id]["target"]

    if target == 1:
        st.error("🦠 Pneumonia")
    else:
        st.success("✅ Normal")

    st.markdown("---")

    st.subheader("Image Metadata")

    st.write(f"**Rows:** {metadata['rows']}")
    st.write(f"**Columns:** {metadata['columns']}")
    st.write(f"**Modality:** {metadata['modality']}")
    st.write(f"**Pixel Spacing:** {metadata['pixel_spacing']}")
    st.write(
        f"**Photometric Interpretation:** {metadata['photometric_interpretation']}"
    )

st.divider()