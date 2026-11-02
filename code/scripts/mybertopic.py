import numpy as np
import pandas as pd
import umap
import hdbscan
import altair as alt

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MinMaxScaler

from sentence_transformers import SentenceTransformer

class MyBERTopic:
    def __init__(
        self,
        sentence_transformer_name: str = "all-MiniLM-L6-v2",
        umap_model=None,
        hdbscan_model=None,
        vectorizer_model=None,
        weight_scheme=None,
        verbose: bool = False,
        seed : int = 92
    ):
        self.verbose = verbose
        self.seed = seed

        # --- Components ---
        self.embedder = SentenceTransformer(sentence_transformer_name)

        self.reducer = umap_model or umap.UMAP(
            n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=self.seed
        )

        self.clusterer = hdbscan_model or hdbscan.HDBSCAN(
            min_cluster_size=10, metric="euclidean", cluster_selection_method="eom"
        )

        self.vectorizer = vectorizer_model or CountVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=2
        )

        self.weight_scheme = weight_scheme or TfidfTransformer()

        # Storage
        self.documents_ = None
        self.embeddings_ = None
        self.reduced_embeddings_ = None
        self.topics_ = None
        self.topic_words_ = None
        self.topic_words_weights_ = None
        self.doc_topic_df_ = None

    # =======================================================
    #  Core Pipeline
    # =======================================================

    def _embed(self, documents: list[str]):
        if self.verbose:
            print("Embedding documents ...")
        return self.embedder.encode(documents, show_progress_bar=self.verbose)

    def _reduce(self, embeddings: np.ndarray):
        if self.verbose:
            print("Reducing embeddings ...")
        return self.reducer.fit_transform(embeddings)

    def _cluster(self, reduced_embeddings: np.ndarray):
        if self.verbose:
            print("Clustering reduced embeddings ...")
        return self.clusterer.fit_predict(reduced_embeddings)

    def _tokenize(self, documents:list[str], topics: np.ndarray):
        if self.verbose:
            print("Tokenizing clustered texts...")

        dict_topics = { # join all document within topic (sorted)
            topic:" ".join(np.array(documents)[np.array(topics)==topic])
            for topic in sorted(list(set(topics)))
        }

        X = self.vectorizer.fit_transform(dict_topics.values())
        words = np.array(self.vectorizer.get_feature_names_out())
        return X, words

    def _weight(self,topics:np.ndarray, X:np.ndarray, words:np.ndarray, n_words:int=10):
        if self.verbose:
            print("Extracting topic words...")
        weights = self.weight_scheme.fit_transform(X)
        ordered_topics = sorted(list(set(topics)))
        topic_words_dict = {
            ordered_topics[i]: words[np.argsort(weight_vector.toarray()[0])[::-1][:n_words]] # topic : words[top indices]
            for i, weight_vector in enumerate(weights)
        }
        topic_words_weights_dict = {
            ordered_topics[i]: np.sort(weight_vector.toarray()[0])[::-1][:n_words]
            for i, weight_vector in enumerate(weights)
        }
        return topic_words_dict, topic_words_weights_dict

    def _extract_topic_words(self, documents: list[str], topics: np.ndarray):
        """Compute topic keywords using a simplified c-TF-IDF approach."""
        X, words = self._tokenize(documents, topics)
        topic_words_dict, topic_words_weights_dict = self._weight(topics=topics, X=X, words=words)

        return topic_words_dict, topic_words_weights_dict

    # =======================================================
    #  Fit / Transform
    # =======================================================

    def fit(self, documents: list[str]):
        self.documents_ = documents
        self.embeddings_ = self._embed(documents)
        self.reduced_embeddings_ = self._reduce(self.embeddings_)
        self.topics_ = self._cluster(self.reduced_embeddings_)
        self.topic_words_, self.topic_words_weights_ = self._extract_topic_words(documents, self.topics_)
        self._build_doc_topic_df()
        return self

    def transform(self, new_documents: list[str]):
        """Assign topics to new docs based on nearest cluster (simple heuristic)."""
        new_embeddings = self._embed(new_documents)
        reduced = self.reducer.transform(new_embeddings)
        labels, strengths = self.clusterer.approximate_predict(reduced)
        return labels, strengths

    def fit_transform(self, documents: list[str]):
        self.fit(documents)
        return self.topics_

    # =======================================================
    #  Helper Methods
    # =======================================================

    def _build_doc_topic_df(self):
        self.doc_topic_df_ = pd.DataFrame({
            "document": self.documents_,
            "topic": self.topics_
        })
        return self.doc_topic_df_

    # =======================================================
    #  Visualization Utilities
    # =======================================================

    def visualise_embeddings(
        self, 
        filter_out_noise=False,
        metadata:dict=None,
    ):
        df = pd.DataFrame({
            "Dim_1": self.reduced_embeddings_[:, 0],
            "Dim_2": self.reduced_embeddings_[:, 1],
            "cluster": self.topics_,
            "text": self.documents_,
        })
        
        # if provided metadata: add it to df + tooltip
        tooltip_vars = ["text", "cluster"]
        if not metadata is None:
            for k, v in metadata.items():
                df[k] = v
                tooltip_vars += [k]

        if filter_out_noise:
            df = df[df.cluster != -1]

        chart = (
            alt.Chart(df)
            .mark_circle(size=100)
            .encode(
                x="Dim_1",
                y="Dim_2",
                color="cluster:N",
                tooltip=tooltip_vars[::-1],
            )
            .interactive()
            .properties(width=600, height=500, title="Embedding Clusters")
        )
        return chart

    def visualize_barchart(self, top_n: int = 10):
        if self.topic_words_ is None or self.topic_words_weights_ is None:
            raise ValueError("Model must be fitted before visualization.")

        # 1. Build tidy dataframe
        records = []
        for topic, words in self.topic_words_.items():
            weights = np.array(self.topic_words_weights_[topic]).flatten()
            top_indices = np.argsort(weights)[::-1][:top_n]
            for i in top_indices:
                records.append({
                    "topic": topic,
                    "word": words[i],
                    "weight": float(weights[i])
                })

        df = pd.DataFrame(records)

        # Remove noise topic if exists (-1)
        df = df[df["topic"] != -1]

        # 2. Define chart
        base = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("weight:Q", title="Word Weight", scale=alt.Scale(domain=(0, df["weight"].max() * 1.1))),
                y=alt.Y("word:N", sort="-x", title=None),
                tooltip=["topic:N", "word:N", alt.Tooltip("weight:Q", format=".4f")],
                color=alt.Color("topic:N", legend=None),
            )
            .properties(width=300, height=200)
        )

        # 3. Make facets per topic
        chart = (
            base
            .facet(
                facet=alt.Facet("topic:N", title="Topic"),
                columns=3  # adjust layout
            )
            .resolve_scale(x="independent", y="independent")
            .properties(title=f"Top {top_n} Words per Topic")
        )

        return chart


    # =======================================================
    #  Topic-Level Visualization
    # =======================================================

    def visualize_topics(
        self, 
        top_n_words: int = 5,
        from_reduced_embeddings: bool = False,
    ):
        """
        Visualize topics in 2D space based on their average embeddings.

        Parameters
        ----------
        top_n_words : int
            Number of top words to display for each topic.
        """
        if self.topics_ is None or self.embeddings_ is None:
            raise ValueError("Model must be fitted before visualization.")

        # Filter out noise (-1)
        valid_idx = self.topics_ != -1
        docs = np.array(self.documents_)[valid_idx]
        topics = self.topics_[valid_idx]

        # --- 1. Compute topic embeddings (average of doc embeddings)
        if from_reduced_embeddings:  
            embeddings = self.reduced_embeddings_[valid_idx]
            embeddings = MinMaxScaler().fit_transform(embeddings)
        else:
            embeddings = self.embeddings_[valid_idx]

        topic_ids = np.unique(topics)
        topic_embeddings = np.array([
            embeddings[topics == t].mean(axis=0) for t in topic_ids
        ])

        # + Compute topics sizes
        topic_sizes = np.array([
            np.sum(np.array(topics)==t) for t in topic_ids
        ])

        # --- 2. Reduce to 2D for visualization IF not using reduced embeddings
        if not from_reduced_embeddings:
            reducer = umap.UMAP(
                n_neighbors=2, n_components=2, metric="cosine", min_dist=0.0, random_state=self.seed
            )
            topic_embeddings = reducer.fit_transform(topic_embeddings)

        # --- 3. Prepare data
        labels = []
        for topic_id in topic_ids:
            words = self.topic_words_.get(topic_id, [])
            top_words = ", ".join(words[:top_n_words])
            labels.append(top_words)

        df_topics = pd.DataFrame({
            "topic": topic_ids,
            "Dim_1": topic_embeddings[:, 0],
            "Dim_2": topic_embeddings[:, 1],
            "top_words": labels,
            "size":topic_sizes,
        })

        # --- 4. Create chart
        chart = (
            alt.Chart(df_topics)
            .mark_circle(opacity=0.7)
            .encode(
                x=alt.X("Dim_1", title="Topic Dim 1", scale=alt.Scale(zero=False)),
                y=alt.Y("Dim_2", title="Topic Dim 2", scale=alt.Scale(zero=False)),
                color="topic:N",
                size=alt.Size("size:Q",
                              title="Documents per Topic",
                              scale=alt.Scale(range=[100, 2000])),  # adjust bubble size
                tooltip=["topic", "top_words", "size"],
            )
            .interactive()
            .properties(width=600, height=500, title="Topic Map")
        )

        text = (
            alt.Chart(df_topics)
            .mark_text(align="center", baseline="middle", dy=-15, fontSize=11)
            .encode(x="Dim_1", y="Dim_2", text="topic")
        )

        return chart + text

    # =======================================================
    #  BONUS!
    # =======================================================

    def find_similar_documents(self, query: str, top_n: int = 5):
        if self.embeddings_ is None or self.documents_ is None:
            raise ValueError("Model must be fitted before searching for similar documents.")

        # 1. Embed query
        query_embedding = self._embed([query])[0]

        # 2. Compute cosine similarity 
        doc_embeddings = self.embeddings_
        similarities = np.dot(doc_embeddings, query_embedding) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # 3. Get top n results
        top_indices = np.argsort(similarities)[::-1][:top_n]

        results = pd.DataFrame({
            "document": np.array(self.documents_)[top_indices],
            "similarity": similarities[top_indices],
            "topic": np.array(self.topics_)[top_indices],
        }).reset_index(drop=True)

        return results
