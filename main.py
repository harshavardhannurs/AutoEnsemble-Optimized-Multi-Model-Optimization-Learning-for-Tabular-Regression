from pathlib import Path

import streamlit as st

from models.ffn import ffn_pipelined_model
from utils import model_utils as mu


DATA_DIR = Path(__file__).resolve().parent / "data"
datasets = sorted(
    path.name
    for path in DATA_DIR.iterdir()
    if path.is_dir() and list(path.glob("*/y_test.parquet"))
)

st.set_page_config(page_title="AutoML Model Runner", page_icon="⚡")
st.title("AutoML Model Runner")

dataset_name = st.selectbox(
    "Choose a dataset",
    datasets,
    format_func=lambda name: name.replace("_", " ").title(),
)

if st.button("Run pipeline", type="primary"):
    try:
        with st.spinner("Training the model..."):
            train_losses, val_losses, r2_scores = ffn_pipelined_model(
                dataset_name=dataset_name,
                batch_size=64,
                no_of_layers=2,
                no_of_units=128,
                learning_rate=0.001,
                weight_decay=0.001,
                batch_norm=False,
                num_epochs=50,
                n_splits=5,
            )

        st.success("Pipeline completed.")
        st.metric("Average R² score", f"{mu.average_test_loss(r2_scores):.4f}")
        st.pyplot(mu.plot_losses(train_losses, val_losses))
    except Exception as error:
        st.error(f"Pipeline failed: {error}")
