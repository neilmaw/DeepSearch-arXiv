import requests
from bs4 import BeautifulSoup
import pickle
import os
class Crawler:
    def __init__(self):
        pass

    def crawl_arxiv(self, query="Graph Neural Networks", max_papers=50):
        url = f"https://arxiv.org/search/?query={query.replace(' ', '+')}&searchtype=all&abstracts=show&order=-announced_date_first&size={max_papers}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        print(f"Fetching {max_papers} papers: {url} - Status: {response.status_code}")

        soup = BeautifulSoup(response.text, "html.parser")
        papers = []
        for item in soup.select("li.arxiv-result")[:max_papers]:
            title = item.select_one("p.title").get_text(strip=True, separator=' ')
            abstract = item.select_one("span.abstract-full").get_text(strip=True, separator=' ').replace("▽ More", "").replace("△ Less",
                                                                                                                "")
            url_elem = item.select_one("a[href*='/abs/']")
            paper_url = f"{url_elem['href']}" if url_elem else ""
            author_elem = item.select_one("p.authors")
            author = author_elem.get_text(strip=True, separator=' ') if author_elem else ""
            papers.append({"title": title, "abstract": abstract, "url": paper_url, "author": author})

        print("Sample papers fetched:")
        for p in papers[:5]:
            print(f"Title: {p['title']}")
            print(f"URL: {p['url']}")
            print(f"Abstract (first 100 chars): {p['abstract'][:100]}")
            print(f"Author: {p['author']}")

        #save
        folder_path = 'data'
        file_path = os.path.join(folder_path, 'papers.pkl')

        # Create folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(papers, f)
        return papers

if __name__ == "__main__":
    keywords = "quantum computing"
    crawler = Crawler()
    crawler.crawl_arxiv(keywords)

