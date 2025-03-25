# main.py
import json
from pyspark.sql import SparkSession
from inputprocessor import InputProcessor
from crawler import Crawler
from invertedindex import InvertedIndex
from BERTEmbedding import BertEmbeddingBuilder
from searchranker import PaperSearchRanker
from arvixRAG import PaperSummarizer
from grokClient import GrokClient

class Main:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("app") \
            .master("local[*]") \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")

    def run(self, question, q):
        msg = ["Checking input question...\n"]
        self.processor = InputProcessor(msg)
        question, keywords = self.processor.processInput(question)
        q.put({"step": "".join(msg)})
        q.put({"step": "Looking Up arXiv..."})

        self.crawler = Crawler(q)
        self.crawler.crawl_arxiv(keywords)
        q.put({"step": "──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────"})
        q.put({"step": "──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────"})
        q.put({"step": "Building inverted index..."})
        self.it = InvertedIndex(q, spark=self.spark)
        self.it.build()
        q.put({"step": "──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────"})
        q.put({"step": "──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────"})

        q.put({"step": "Building embeddings..."})
        self.builder = BertEmbeddingBuilder(q, spark=self.spark)
        self.builder.build_embeddings(text_field="abstract")
        q.put({"step": "──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────"})
        q.put({"step": "──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────"})

        q.put({"step": "Ranking papers..."})
        self.ranker = PaperSearchRanker(q, spark=self.spark)
        papers = self.ranker.search_and_rank(question, 5)
        q.put({"step": "──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────"})
        q.put({"step": "──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────"})

        q.put({"step": "Summarizing papers..."})
        self.summarizer = PaperSummarizer(q)
        summaries = self.summarizer.process_topk_papers(papers)
        q.put({"step": "──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────"})
        q.put({"step": "──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────"})

        q.put({"step": "Querying Grok..."})
        self.client = GrokClient(q)
        response = self.client.request(summaries, question)
        q.put({"step": "──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────"})
        q.put({"step": "──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────"})

        q.put({"result": response})
        q.put({"final": True})  # Signals end of stream

