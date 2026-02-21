import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx

from pipeline.preprocess import clean_dataframe
from pipeline.topic_model import run_lda, TOPIC_LABELS, N_TOPICS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="ClinsightAI", layout="wide")
st.title("🏥 ClinsightAI — Healthcare Review Intelligence")

# ── Run pipeline (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df_raw = pd.read_csv("hospital.csv")
    df = clean_dataframe(df_raw, text_col='Feedback')
    df, topic_probabilities, topic_words, theme_counts, theme_freq, severity_df, topic_similarity = run_lda(df)
    return df, topic_probabilities, topic_words, theme_counts, theme_freq, severity_df, topic_similarity

with st.spinner("Running analysis on hospital reviews..."):
    df, topic_probabilities, topic_words, theme_counts, theme_freq, severity_df, topic_similarity = load_data()

st.success(f"✅ Analysed {len(df):,} reviews")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Theme Discovery", "⚠️ Severity & Risk"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Theme Discovery
# ════════════════════════════════════════════════════════════════════════════
with tab1:

    st.subheader("Theme Frequency Distribution")
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        sns.barplot(x=theme_freq.values, y=theme_freq.index, palette="Blues_d", ax=ax1)
        ax1.set_xlabel("Frequency (proportion of reviews)")
        ax1.set_ylabel("")
        ax1.set_title("Theme Frequency")
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.pie(theme_counts.values, labels=theme_counts.index,
                autopct='%1.1f%%', startangle=140)
        ax2.set_title("Theme Distribution")
        st.pyplot(fig2)
        plt.close(fig2)

    st.divider()

    st.subheader("Top Words per Topic")
    cols = st.columns(N_TOPICS)
    for i, (theme, words) in enumerate(topic_words.items()):
        with cols[i]:
            st.markdown(f"**{theme}**")
            for w in words[:10]:
                st.markdown(f"- {w}")

    st.divider()

    st.subheader("Topic Similarity Network (Cosine Similarity)")
    G          = nx.Graph()
    labels_list = list(TOPIC_LABELS.values())
    for label in labels_list:
        G.add_node(label)
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            G.add_edge(labels_list[i], labels_list[j], weight=topic_similarity[i][j])

    pos     = nx.spring_layout(G, seed=42)
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    fig3, ax3 = plt.subplots(figsize=(10, 7))
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', ax=ax3)
    nx.draw_networkx_edges(G, pos, width=[w * 5 for w in weights], edge_color='gray', ax=ax3)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax3)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax3)
    ax3.set_title("Topic Similarity Network")
    ax3.axis('off')
    st.pyplot(fig3)
    plt.close(fig3)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Severity & Risk
# ════════════════════════════════════════════════════════════════════════════
with tab2:

    st.subheader("Theme Severity Scores")
    st.dataframe(severity_df, use_container_width=True)

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    heatmap_data = severity_df.set_index("Theme")[["Severity Score"]]
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="Reds", ax=ax4)
    ax4.set_title("Theme Severity Heatmap")
    st.pyplot(fig4)
    plt.close(fig4)

    st.divider()

    st.subheader("Severity Score by Theme")
    fig5, ax5 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=severity_df, x="Severity Score", y="Theme", palette="Reds_d", ax=ax5)
    ax5.set_title("Severity Score (Frequency × Avg Probability)")
    st.pyplot(fig5)
    plt.close(fig5)