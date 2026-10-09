
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
EXPLAIN_DIR = ROOT / "outputs" / "explainability"
EVAL_DIR = EXPLAIN_DIR / "quantitative_evaluation"
GALLERY_DIR = EXPLAIN_DIR / "gallery"

st.set_page_config(
    page_title="AstraMed | Explainability",
    page_icon="🩻",
    layout="wide",
)

st.title("AstraMed — Explainability Dashboard")
st.caption(
    "Research prototype • RSNA pneumonia dataset • "
    "Explanations are not clinically validated."
)


@st.cache_data
def load_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_json(path):
    import json
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


per_patient = load_csv(EVAL_DIR / "per_patient_metrics.csv")
aggregate = load_csv(EVAL_DIR / "aggregate_metrics.csv")
thresholds = load_csv(EVAL_DIR / "threshold_sensitivity.csv")
metadata = load_json(EVAL_DIR / "evaluation_metadata.json")

tab_patient, tab_cohort, tab_report = st.tabs(
    ["Patient explorer", "Cohort evaluation", "Sprint 6 report"]
)

with tab_patient:
    st.subheader("Patient-level explanations")

    if per_patient.empty:
        st.warning(
            "M6 results are missing. Run the quantitative evaluation first."
        )
    else:
        patient_ids = sorted(per_patient["patient_id"].dropna().unique())
        patient_id = st.selectbox("Select patient", patient_ids)

        method_options = {
            "ResNet-50 Grad-CAM": "resnet50_gradcam",
            "ViT-B/16 attention": "vit_b16_attention",
        }
        selected_label = st.radio(
            "Explanation method",
            list(method_options),
            horizontal=True,
        )
        method = method_options[selected_label]

        rows = per_patient[
            (per_patient["patient_id"] == patient_id)
            & (per_patient["method"] == method)
        ]

        if rows.empty:
            st.info("No metrics were found for this patient and method.")
        else:
            row = rows.iloc[0]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pneumonia probability", f"{row['pneumonia_probability']:.3f}")
            c2.metric("IoU", f"{row['iou']:.3f}")
            c3.metric("F1", f"{row['f1']:.3f}")
            c4.metric("Pointing-game hit", "Yes" if bool(row["pointing_game_hit"]) else "No")

        image_path = GALLERY_DIR / f"{patient_id}_{method}.png"
        if image_path.exists():
            st.image(
                str(image_path),
                caption=f"{selected_label} — patient {patient_id}",
                use_container_width=True,
            )
        else:
            st.warning(
                "Gallery image not generated for this patient. "
                "Run scripts/build_explainability_gallery.py."
            )

        st.caption(
            "The heatmap shows relative explanation intensity. "
            "Overlap with a box is not proof of clinical correctness."
        )


with tab_cohort:
    st.subheader("Quantitative comparison")

    if aggregate.empty:
        st.warning("Aggregate metrics are missing. Run M6 first.")
    else:
        st.markdown("### Metrics at the 80th-percentile threshold")

        primary = aggregate[
            aggregate["threshold_percentile"] == 80
        ].copy()

        if not primary.empty:
            columns = [
                col for col in [
                    "method",
                    "patient_count",
                    "iou_mean",
                    "iou_std",
                    "precision_mean",
                    "recall_mean",
                    "f1_mean",
                    "pointing_game_hit_mean",
                    "energy_inside_boxes_mean",
                ] if col in primary.columns
            ]
            st.dataframe(
                primary[columns],
                use_container_width=True,
                hide_index=True,
            )

            if "iou_mean" in primary.columns:
                chart_data = primary.set_index("method")[["iou_mean"]]
                st.bar_chart(chart_data)

        st.markdown("### Threshold sensitivity")
        if not thresholds.empty:
            method_filter = st.multiselect(
                "Methods",
                sorted(thresholds["method"].unique()),
                default=sorted(thresholds["method"].unique()),
            )
            filtered = thresholds[
                thresholds["method"].isin(method_filter)
            ]

            if not filtered.empty:
                summary = (
                    filtered.groupby(["method", "threshold_percentile"])
                    .agg(
                        mean_iou=("iou", "mean"),
                        mean_f1=("f1", "mean"),
                    )
                    .reset_index()
                )
                st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown("### Evaluation metadata")
        st.json(metadata)


with tab_report:
    st.subheader("Sprint 6 summary")

    report_path = EXPLAIN_DIR / "sprint6_report.md"
    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))
    else:
        st.info(
            "Sprint 6 report not generated yet. "
            "Run scripts/generate_sprint6_report.py."
        )

    comparison_path = EVAL_DIR / "method_comparison.md"
    if comparison_path.exists():
        with st.expander("M6 method-comparison report"):
            st.markdown(comparison_path.read_text(encoding="utf-8"))
