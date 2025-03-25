# ArXiv Deep Search Assistant

![Status](https://img.shields.io/badge/Status-Active-brightgreen)  
![Python](https://img.shields.io/badge/Python-3.9+-yellow)

[deepsearch-arxiv](http://deepsearch-arxiv.com/) is a powerful AI-driven tool to explore and summarize scientific papers from arXiv, leveraging Retrieval-Augmented Generation (RAG) and Grok (by xAI) for insightful, context-aware answers.

## Overview

The **ArXiv Deep Search Assistant** empowers researchers, students, and enthusiasts to seamlessly explore scientific literature on arXiv. This tool automates the process of crawling arXiv for relevant papers, builds a Retrieval-Augmented Generation (RAG) system using BERT embeddings and inverted indexing, and integrates with Grok (by xAI) to deliver detailed, context-aware responses to your queries.
## Features

- **Automated arXiv Crawling**: Fetches relevant papers from arXiv based on your query.
- **RAG System**: Utilizes BERT embeddings and inverted indexing for efficient paper retrieval.
- **AI-Powered Summarization**: Employs Grok (by xAI) to generate detailed, context-aware answers and summaries.
- **Real-Time Streaming**: Streams results via WebSocket for a seamless user experience.

## Prerequisites

- **Python Version**: Requires **Python 3.9** to avoid potential version conflicts.
- **EC2 Instance (Optional)**: For hosting, an AWS EC2 instance with at least 15 GiB RAM and 2 GiB swap is recommended. Currently utilizes CPU for cost effectiveness.
- **Dependencies**: Listed in `requirements.txt`.
- **Java (for PySpark)**: Amazon Corretto 11 is required if using PySpark (currently bypassed for debugging).

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/neilmaw/DeepSearch-arXiv.git
   cd DeepSearch-arXiv
   ```

2. **Set Up a Virtual Environment**:
  ```bash
   python3.9 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
  ```bash
   pip install -r requirements.txt
   ```

4. **Test Your App**:
  ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

## Acknowledgments

- Built with ❤️ by [neilmaw](https://github.com/neilmaw).
- Powered by [arXiv](https://arxiv.org/) and [Grok](https://xai.ai/) (by xAI).


