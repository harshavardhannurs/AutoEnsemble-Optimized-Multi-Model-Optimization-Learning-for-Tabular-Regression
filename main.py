from pathlib import Path
import re

import streamlit as st

from smac_utils import optimize_models


DATA_DIR = Path(__file__).resolve().parent / "data"
datasets = sorted(
    path.name
    for path in DATA_DIR.iterdir()
    if path.is_dir() and list(path.glob("*/y_test.parquet"))
)

st.set_page_config(page_title="AutoML with SMAC", page_icon="⚡")
st.title("AutoML Model Selection with SMAC")

dataset = st.selectbox(
    "Choose a dataset",
    datasets,
    format_func=lambda name: name.replace("_", " ").title(),
)

trials = st.number_input(
    "SMAC trials",
    min_value=6,
    value=10,
    help="Stage 1 evaluates all six model families; Stage 2 builds the ensemble.",
)
folds = st.number_input("Cross-validation folds", min_value=2, value=5)

if st.button("Find best hyperparameters", type="primary"):
    log_messages = []
    live_log = st.empty()
    stage_column, trial_column, fold_column = st.columns(3)
    stage_display = stage_column.empty()
    trial_display = trial_column.empty()
    fold_display = fold_column.empty()
    stage_display.metric("Stage", "Stage 1")
    trial_display.metric("Trial", f"0/{int(trials)}")
    fold_display.metric("Fold", f"0/{int(folds)}")

    def update_log(message):
        log_messages.append(message)

        trial_match = re.search(r"Trial (\d+)/(\d+) started", message)
        fold_match = re.search(r"CV Fold: (\d+)/(\d+)", message)
        if trial_match:
            stage_display.metric("Stage", "Stage 1")
            trial_display.metric("Trial", f"{trial_match.group(1)}/{trial_match.group(2)}")
            fold_display.metric("Fold", f"0/{int(folds)}")
        if message == "Stage 2 ensemble started":
            stage_display.metric("Stage", "Stage 2")
            trial_display.metric("Trial", f"{int(trials)}/{int(trials)}")
            fold_display.metric("Fold", f"0/{int(folds)}")
        if fold_match:
            fold_display.metric("Fold", f"{fold_match.group(1)}/{fold_match.group(2)}")

        epoch_message = message.startswith(("Epoch", "Ensemble FFN Epoch"))
        if not epoch_message or len(log_messages) % 10 == 0:
            live_log.code("\n".join(log_messages[-30:]), language=None)

    try:
        with st.spinner("SMAC is testing model configurations..."):
            best_config, best_r2 = optimize_models(
                dataset,
                int(trials),
                int(folds),
                progress_callback=update_log,
            )

        live_log.empty()
        stage_display.metric("Stage", "Complete")
        st.success("Optimization completed.")
        st.metric("Best mean R²", f"{best_r2:.4f}")
        st.subheader("Best hyperparameters")
        st.json(best_config)
    except Exception as error:
        stage_display.metric("Stage", "Failed")
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
