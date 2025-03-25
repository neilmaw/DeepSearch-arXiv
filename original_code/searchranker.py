from sentence_transformers import SentenceTransformer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, regexp_extract
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
import ssl



class PaperSearchRanker:
    def __init__(self, embeddings_path="data/bert_embeddings.parquet", index_path="data/inverted_index.parquet",
                 model_name="all-MiniLM-L6-v2"):
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        nltk.download('punkt')
        self.embeddings_path = embeddings_path
        self.index_path = index_path
        self.model = SentenceTransformer(model_name)
        self.spark = SparkSession.builder.appName("PaperSearchRanker").master("local[*]").getOrCreate()
        self.embeddings_df = self.spark.read.parquet(self.embeddings_path)
        self.inverted_index = self.spark.read.parquet(self.index_path)

    def cosine_similarity(self, embedding1, embedding2):
        arr1, arr2 = np.array(embedding1), np.array(embedding2)
        return float(np.dot(arr1, arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2)))

    def search_and_rank(self, query, top_k=3):
        query_tokens = word_tokenize(query.lower())
        print(f"Query tokens: {query_tokens}")

        # Retrieve candidates
        candidates = self.inverted_index.filter(col("token").isin(query_tokens)) \
            .select(explode("urls").alias("url")).distinct()
        print(f"Candidate papers: {candidates.count()}")

        # Join with embeddings
        embeddings_df = self.embeddings_df.withColumn("url_id", regexp_extract(col("url"), r"(\d+\.\d+)", 1))

        candidates_df = candidates.withColumn("url_id", col("url")).join(embeddings_df, "url_id", "inner")
        print(f"Candidates with embeddings: {candidates_df.count()}")

        if candidates_df.count() == 0:
            return []

        # Collect candidates to driver and rank locally
        candidates = candidates_df.collect()
        query_embedding = self.model.encode([query])[0].tolist()
        ranked_candidates = sorted(candidates, key=lambda x: self.cosine_similarity(x["embedding"], query_embedding),
                                   reverse=True)[:top_k]

        # Format top-k papers
        top_k_papers = [
            {"title": row["title"], "url": row["url"], "abstract": row["abstract"], "author": row["author"]}
            for row in ranked_candidates
        ]

        print(f"Top {top_k} papers for '{query}':")
        for paper in top_k_papers:
            print(f"Title: {paper['title']}, URL: {paper['url']}")
        return top_k_papers

    def cleanup(self):
        self.spark.stop()
        print("Spark session stopped.")


if __name__ == "__main__":
    ranker = PaperSearchRanker()
    query = "what is robotics computer vision"
    papers = ranker.search_and_rank(query)
    ranker.cleanup()