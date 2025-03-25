import requests
import pdfplumber
import io
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
class PaperSummarizer:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        model_name = "philschmid/bart-large-cnn-samsum"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        # Move to CPU explicitly for EC2
        self.model.to("cpu")
        self.model.eval()  # Set to evaluation mode

    def get_pdf_text(self, paper_id, max_pages=8):
        """Fetch and extract text from the PDF using paper ID."""
        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"  # Construct full PDF URL from ID
        try:
            response = requests.get(pdf_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                text = " ".join(page.extract_text() or "" for page in pdf.pages[:max_pages])
            return text.strip()[:10000]
        except Exception as e:
            print(f"Error fetching PDF for {paper_id}: {e}")
            return ""

    def summarize_paper(self, text, max_length=200, min_length=150):
        """Summarize the extracted text."""
        if not text:
            return "No summary available."
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",  # PyTorch tensors
                truncation=True,
                padding=True
            ).to("cpu")  # Ensure inputs are on CPU

            # Generate summary
            with torch.no_grad():  # Disable gradient computation
                summary_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,  # Deterministic output
                    num_beams=4,      # Beam search for better quality
                    early_stopping=True
                )

            # Decode summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            print(f"Error summarizing: {e}")
            return text[:200] + "..."
    def process_topk_papers(self, top_k_papers):
        """Process top-k papers with IDs and return summaries."""
        summaries = []
        for paper in top_k_papers:
            paper_id = paper["url"]  # URL is now just the ID
            print(f"Processing: {paper['title']} ({paper_id})")
            text = self.get_pdf_text(paper_id)
            summary = self.summarize_paper(text)
            summaries.append({
                "title": paper["title"],
                "url": f"https://arxiv.org/abs/{paper_id}",  # Full URL for output
                "summary": summary
            })
        return summaries

# Example usage
if __name__ == "__main__":
    from searchranker import PaperSearchRanker

    ranker = PaperSearchRanker()
    query = "what is robot surgeon"
    top_k_papers = ranker.search_and_rank(query, top_k=3)
    ranker.cleanup()

    summarizer = PaperSummarizer()
    summaries = summarizer.process_topk_papers(top_k_papers)
    for s in summaries:
        print(f"\nTitle: {s['title']}")
        print(f"URL: {s['url']}")
        print(f"Summary: {s['summary']}")