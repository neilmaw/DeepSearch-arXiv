import pickle
from urllib.parse import urlparse
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import nltk
import ssl

class InvertedIndex:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("MiniGrokInvertedIndex") \
            .master("local[*]") \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")

        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Error with NLTK: {e}")
            exit(1)

    def extract_arxiv_id(self, url):
        if not url:
            return None
        path = urlparse(url).path
        match = re.search(r'(\d{4}\.\d{5})', path)
        return match.group(1) if match else None

    def tokenize_text(self, text):
        if not text:
            return []
        text = text.lower()
        #gather contents from parenthesis such as "(gnn)"
        paren_matches = re.findall(r'\(([^()]*)\)', text)
        #remove parenthesis content
        main_text = re.sub(r'\([^()]*\)', '', text)
        #remove punctuation
        main_text = re.sub(r'[^\w\s]', '', main_text)
        tokens = word_tokenize(main_text)
        for match in paren_matches:
            match_clean = re.sub(r'[^\w\s]', '', match)
            tokens.extend(word_tokenize(match_clean))
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        return tokens
    def build(self, testkeywords=[]):
        with open("data/papers.pkl", "rb") as f:
            papers = pickle.load(f)

        # Add ArXiv ID and tokens
        for paper in papers:
            paper["arxiv_id"] = self.extract_arxiv_id(paper["url"])
            paper["title_tokens"] = self.tokenize_text(paper["title"])
            paper["abstract_tokens"] = self.tokenize_text(paper["abstract"])

        # Debug tokenization
        print("\nSample tokenized papers:")
        for p in papers[:5]:
            print(f"Title: {p['title']}")
            print(f"Title Tokens: {p['title_tokens']}")
            print(f"Abstract Tokens (first 10): {p['abstract_tokens'][:10]}")
            print(f"ArXiv ID: {p['arxiv_id']}")

        # Create Spark DataFrame
        papers_rdd = self.spark.sparkContext.parallelize(papers)
        papers_df = self.spark.createDataFrame(papers_rdd)
        print("\nPapers DataFrame sample:")
        papers_df.select("arxiv_id", "title", "title_tokens", "abstract_tokens").show(5, truncate=False)

        # Build inverted index from title and abstract tokens
        token_to_arxiv_rdd = papers_rdd.flatMap(
            lambda paper: [(token, paper["arxiv_id"]) for token in paper["title_tokens"] + paper["abstract_tokens"]]
        )
        inverted_index_rdd = token_to_arxiv_rdd.groupByKey().mapValues(list)
        inverted_index_df = self.spark.createDataFrame(
            inverted_index_rdd,
            schema="token string, urls array<string>"
        )

        # Debug inverted index
        print("\nInverted Index sample:")
        if len(testkeywords) == 0:
            inverted_index_df.orderBy("token").show(5, truncate=False)
        else:
            inverted_index_df.filter(inverted_index_df.token.isin(testkeywords)).show(5, truncate=False)

        # Save outputs
        inverted_index_df.write.mode("overwrite").parquet("data/inverted_index.parquet")
        papers_df.write.mode("overwrite").parquet("data/papers.parquet")
        # print("Saved inverted_index.parquet and papers.parquet")

        self.spark.stop()

if __name__ == "__main__":
    invertedindex = InvertedIndex()
    invertedindex.build()
    # text = "We propose a natural quantization of a standard neural network, where the neurons correspond to qubits and the activation functions are implemented via quantum gates and measurements. The simplest quantized neural network corresponds to applying single-qubit rotations, with the rotation angles being dependent on the weights and measurement outcomes of the previous layer. This realization has the advantage of being smoothly tunable from the purely classical limit with no quantum uncertainty (thereby reproducing the classical neural network exactly) to a quantum case, where superpositions introduce an intrinsic uncertainty in the network. We benchmark this architecture on a subset of the standard MNIST dataset and find a regime of \"quantum advantage,\" where the validation error rate in the quantum realization is smaller than that in the classical model. We also consider another approach where quantumness is introduced via weak measurements of ancilla qubits entangled with the neuron qubits. This quantum neural network also allows for smooth tuning of the degree of quantumness by controlling an entanglement angle, g, with g=π2 replicating the classical regime. We find that validation error is also minimized within the quantum regime in this approach. We also observe a quantum transition, with sharp loss of the quantum network's ability to learn at a critical point gc. The proposed quantum neural networks are readily realizable in present-day quantum computers on commercial datasets."
    # tokens = invertedindex.tokenize_text(text)
    # print(tokens)