import streamlit as st
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
from models.rag_pipeline import RAGSystem
from evaluation.metrics import compute_binary_metrics, hallucination_rate

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="LLM RAG Research Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0b1120;
}
.title-gradient {
    font-size: 48px;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #a78bfa, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    color: #94a3b8;
    font-size: 18px;
}
.glass-card {
    background: rgba(30,41,59,0.6);
    backdrop-filter: blur(12px);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.6);
}
.metric-big {
    font-size: 36px;
    font-weight: bold;
}
.section-space {
    margin-top: 60px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🔬 Research Lab Control Panel")
st.sidebar.markdown("""
**Experiment Configuration**
- Architecture: RAG
- Embedding Model: nomic-embed-text
- Evaluation: Precision / Recall / F1
- Hallucination Metric: Lexical Overlap
""")

# ---------------- HEADER ----------------
st.markdown('<div class="title-gradient">LLM RAG Experimental Research Lab</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced quantitative benchmarking of Retrieval-Augmented Generation systems.</div>', unsafe_allow_html=True)

st.markdown("<div class='section-space'></div>", unsafe_allow_html=True)

# ---------------- RUN BUTTON ----------------
run = st.button("🚀 Launch Experiment")

if run:

    dataset = pd.read_csv("data/dataset.csv")

    with st.expander("📂 View Dataset Used"):
        st.write(f"Total Samples: {len(dataset)}")
        st.dataframe(dataset)

    questions = dataset["question"].tolist()
    answers = dataset["answer"].tolist()

    models = ["phi3:mini"]

    results = []
    progress = st.progress(0)

    status_placeholder = st.empty()

    for model_name in models:

        status_placeholder.info(f"Running Model: {model_name}")

        system = RAGSystem(model_name)
        system.build_vectorstore("data/dataset.csv")

        rag_preds = []
        latencies = []

        for i, q in enumerate(questions):
            start = time.time()
            rag_resp = system.rag_answer(q)
            latency = time.time() - start

            rag_preds.append(rag_resp)
            latencies.append(latency)

            progress.progress((i + 1) / len(questions))

        y_true = [1] * len(answers)
        y_rag = []

        for expected, predicted in zip(answers, rag_preds):
            expected_words = set(expected.lower().split())
            predicted_words = set(predicted.lower().split())
            overlap = expected_words.intersection(predicted_words)
            y_rag.append(1 if len(overlap) >= 2 else 0)

        precision, recall, f1 = compute_binary_metrics(y_true, y_rag)
        halluc_rate = hallucination_rate(answers, rag_preds)

        results.append({
            "Model": model_name,
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1 Score": round(f1, 3),
            "Hallucination Rate": round(halluc_rate, 3),
            "Avg Latency (sec)": round(sum(latencies) / len(latencies), 3)
        })

    results_df = pd.DataFrame(results)

    status_placeholder.success("Experiment Completed Successfully")

    st.markdown("<div class='section-space'></div>", unsafe_allow_html=True)

    # ---------------- METRIC CARDS ----------------
    st.markdown("## 📊 Performance Overview")

    col1, col2, col3, col4 = st.columns(4)

    for col, metric in zip(
        [col1, col2, col3, col4],
        ["Precision", "Recall", "F1 Score", "Avg Latency (sec)"]
    ):
        with col:
            st.markdown(f"""
            <div class="glass-card">
                <div>{metric}</div>
                <div class="metric-big">{results_df[metric][0]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-space'></div>", unsafe_allow_html=True)

    # ---------------- ADVANCED CHART ----------------
    st.markdown("## 📈 Multi-Metric Comparison")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=["Precision", "Recall", "F1"],
        y=[
            results_df["Precision"][0],
            results_df["Recall"][0],
            results_df["F1 Score"][0]
        ],
        marker_color=["#38bdf8", "#a78bfa", "#22d3ee"]
    ))

    fig.update_layout(
        plot_bgcolor="#0b1120",
        paper_bgcolor="#0b1120",
        font_color="white",
        title="Model Performance Metrics",
        xaxis_title="Metric",
        yaxis_title="Score"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-space'></div>", unsafe_allow_html=True)

    # ---------------- LATENCY GAUGE ----------------
    st.markdown("## ⚡ Latency Analysis")

    latency_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=results_df["Avg Latency (sec)"][0],
        title={'text': "Average Latency (seconds)"},
        gauge={
            'axis': {'range': [0, 20]},
            'bar': {'color': "#38bdf8"}
        }
    ))

    latency_fig.update_layout(
        paper_bgcolor="#0b1120",
        font_color="white"
    )

    st.plotly_chart(latency_fig, use_container_width=True)

    st.download_button(
        label="⬇ Download Full Results",
        data=results_df.to_csv(index=False),
        file_name="research_results.csv",
        mime="text/csv"
    )