import pickle
from sentence_transformers import SentenceTransformer
from pyspark.sql import SparkSession


class BertEmbeddingBuilder:
    def __init__(self, q, spark, pkl_path="data/papers.pkl", embeddings_path="data/bert_embeddings.parquet", model_name="all-MiniLM-L6-v2"):
        """Initialize the BERT embedding builder."""
        self.pkl_path = pkl_path
        self.embeddings_path = embeddings_path
        self.model = SentenceTransformer(model_name)

        self.spark = spark
        self.q = q
        # Load papers
        self.papers = self._load_papers()

    def _load_papers(self):
        """Load papers from pickle file."""
        try:
            with open(self.pkl_path, "rb") as f:
                papers = pickle.load(f)
            # print(f"Loaded {len(papers)} papers from {self.pkl_path}")
            return papers
        except Exception as e:
            print(f"Error loading papers: {e}")
            return []

    def build_embeddings(self, text_field="abstract"):
        """Generate and store BERT embeddings."""
        if not self.papers:
            print("No papers to process.")
            return None

        # Prepare text for embedding
        texts = [paper[text_field] for paper in self.papers]

        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)

        # Combine with metadata
        papers_with_embeddings = [
            {
                "title": paper["title"],
                "abstract": paper["abstract"],
                "url": paper["url"],
                "author": paper["author"],
                "embedding": embedding.tolist()
            }
            for paper, embedding in zip(self.papers, embeddings)
        ]

        # Convert to Spark DataFrame and store
        embeddings_df = self.spark.createDataFrame(papers_with_embeddings)
        embeddings_df.write.mode("overwrite").parquet(self.embeddings_path)
        print(f"Stored {len(papers_with_embeddings)} embeddings in {self.embeddings_path}")
        self.q.put({"step": f"Stored {len(papers_with_embeddings)} embeddings in {self.embeddings_path}"})

        print("sample embeddings: ")
        for sample in papers_with_embeddings[:5]:
            print(sample["title"])
            print(sample["url"])
            print(sample["embedding"])
            self.q.put({"step": sample["title"]})
            self.q.put({"step": sample["url"]})
            self.q.put({"step": sample["embedding"]})
            self.q.put({"step": "======================================="})

        return embeddings_df




# Example usage
if __name__ == "__main__":
    # Initialize the builder
    builder = BertEmbeddingBuilder()

    # Build embeddings
    embeddings_df = builder.build_embeddings(text_field="abstract")

