import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt

vectorizer = CountVectorizer(
    max_df=0.9,
    min_df=5,
    stop_words='english'
)

dtm = vectorizer.fit_transform(df['clean_text'])
lda = LatentDirichletAllocation(
    n_components=5,        # 👈 5 Topics
    random_state=18,
    learning_method='batch'
)

lda.fit(dtm)
def display_topics(model, feature_names, no_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nTopic {topic_idx+1}:")
        print(" | ".join([feature_names[i]
                          for i in topic.argsort()[:-no_top_words - 1:-1]]))

feature_names = vectorizer.get_feature_names_out()
display_topics(lda, feature_names)

topic_probabilities = lda.transform(dtm)

df['dominant_topic'] = topic_probabilities.argmax(axis=1)


topic_labels = {
    0: "Clinical Care Quality",
    1: "Overall Service & Cleanliness",
    2: "Emergency Service Issues",
    3: "Consultation Experience",
    4: "Wait Time & Efficiency"
}

df['theme'] = df['dominant_topic'].map(topic_labels)

theme_counts = df['theme'].value_counts()
theme_freq = theme_counts / len(df)

print("\nTheme Frequency Distribution:")
print(theme_freq)


plt.figure(figsize=(8,5))
sns.barplot(x=theme_freq.values, y=theme_freq.index)
plt.title("Theme Frequency Distribution")
plt.xlabel("Frequency")
plt.ylabel("Theme")
plt.show()


plt.figure(figsize=(6,6))
plt.pie(theme_counts, labels=theme_counts.index, autopct='%1.1f%%')
plt.title("Theme Distribution")
plt.show()


avg_prob = topic_probabilities.mean(axis=0)

severity_scores = theme_freq.values * np.abs(avg_prob)

severity_df = pd.DataFrame({
    "Theme": list(topic_labels.values()),
    "Frequency": theme_freq.values,
    "Avg_Probability": avg_prob,
    "Severity_Score": severity_scores
})

severity_df = severity_df.sort_values(by="Severity_Score", ascending=False)

print(severity_df)

plt.figure(figsize=(8,4))
sns.heatmap(severity_df[['Severity_Score']].set_index(severity_df['Theme']),
            annot=True, cmap="Reds")
plt.title("Theme Severity Heatmap")
plt.show()

from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt

# Compute cosine similarity between topic-word distributions
topic_similarity = cosine_similarity(lda.components_)

# Build graph
G = nx.Graph()

for i in range(len(topic_labels)):
    G.add_node(topic_labels[i])

for i in range(len(topic_labels)):
    for j in range(i+1, len(topic_labels)):
        G.add_edge(topic_labels[i],
                   topic_labels[j],
                   weight=topic_similarity[i][j])

# Position nodes
pos = nx.spring_layout(G, seed=42)

# Extract weights
weights = [G[u][v]['weight'] for u, v in G.edges()]

plt.figure(figsize=(10,8))

# Draw nodes
nx.draw_networkx_nodes(G, pos,
                       node_size=3000,
                       node_color='lightblue')

# Draw edges (scaled for visibility)
nx.draw_networkx_edges(G, pos,
                       width=[w*5 for w in weights],
                       edge_color='gray')

# Draw labels
nx.draw_networkx_labels(G, pos,
                        font_size=10,
                        font_weight='bold')

# 🔥 Draw edge weight labels (THIS IS THE KEY ADDITION)
edge_labels = {(u, v): f"{d['weight']:.2f}"
               for u, v, d in G.edges(data=True)}

nx.draw_networkx_edge_labels(G, pos,
                             edge_labels=edge_labels,
                             font_size=8)

plt.title("Topic Similarity Network (Cosine Similarity)")
plt.axis('off')
plt.show()