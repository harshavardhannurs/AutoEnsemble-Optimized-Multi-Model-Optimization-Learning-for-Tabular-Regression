from pathlib import Path

import streamlit as st

from smac_utils import optimize_ffn


DATA_DIR = Path(__file__).resolve().parent / "data"
datasets = sorted(
    path.name
    for path in DATA_DIR.iterdir()
    if path.is_dir() and list(path.glob("*/y_test.parquet"))
)

st.set_page_config(page_title="AutoML with SMAC", page_icon="⚡")
st.title("AutoML with SMAC")

dataset = st.selectbox(
    "Choose a dataset",
    datasets,
    format_func=lambda name: name.replace("_", " ").title(),
)

trials = st.number_input("SMAC trials", min_value=1, value=10)
folds = st.number_input("Cross-validation folds", min_value=2, value=5)

if st.button("Find best hyperparameters", type="primary"):
    log_messages = []
    live_log = st.empty()

    def update_log(message):
        log_messages.append(message)
        if not message.startswith("Epoch") or len(log_messages) % 10 == 0:
            live_log.code("\n".join(log_messages[-30:]), language=None)

    try:
        with st.spinner("SMAC is testing FFN configurations..."):
            best_config, best_r2 = optimize_ffn(
                dataset,
                int(trials),
                int(folds),
                progress_callback=update_log,
            )

        live_log.empty()
        st.success("Optimization completed.")
        st.metric("Best mean R²", f"{best_r2:.4f}")
        st.subheader("Best hyperparameters")
        st.json(best_config)
    except Exception as error:
        st.error(f"Optimization failed: {error}")

    if log_messages:
        with st.expander("Full optimization log"):
            st.text_area(
                "Training output",
                value="\n".join(log_messages),
                height=400,
                disabled=True,
                label_visibility="collapsed",
            )
