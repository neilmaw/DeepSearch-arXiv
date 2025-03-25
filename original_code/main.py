from inputprocessor import InputProcessor

question = "what is advancements in robot surgeon?"
processor = InputProcessor()
question, keywords = processor.processInput(question)


from crawler import Crawler

crawler = Crawler()
crawler.crawl_arxiv(keywords)


from invertedindex import InvertedIndex

it = InvertedIndex()
it.build(testkeywords=keywords.split(" "))

from BERTEmbedding import BertEmbeddingBuilder

builder = BertEmbeddingBuilder()
embeddings_df = builder.build_embeddings(text_field="abstract")
builder.cleanup()

from searchranker import PaperSearchRanker

ranker = PaperSearchRanker()
papers = ranker.search_and_rank(question, 5)
ranker.cleanup()

from arvixRAG import PaperSummarizer
summarizer = PaperSummarizer()
summaries = summarizer.process_topk_papers(papers)

from grokClient import GrokClient
client = GrokClient()
response = client.request(summaries, question)

print(f"final response: {response}")