# 🔑 Keyword Extraction

Keyword extraction automatically identifies important words or phrases in a text document. It condenses the main topics or themes discussed. Techniques include statistical analysis, NLP algorithms, and machine learning. Widely used in document summarization, SEO, and information retrieval, it aids in organizing and categorizing text data for various applications.

In this project, we used four different techniques for keyword extraction.

## 1. Statistical Methods

These methods rely on statistical properties of words and phrases to identify keywords. Examples include:

### 1.1 TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) weighs words based on their frequency in a document and their rarity across a corpus. This method is effective for highlighting significant terms within a text.

### 1.2 RAKE (Rapid Automatic Keyword Extraction)

RAKE uses a list of stop words and phrase delimiters to identify relevant words and phrases. It ranks keywords based on their frequency and co-occurrence patterns within the text.

### 1.3 YAKE (Yet Another Keyword Extractor)

YAKE is an unsupervised method that relies on statistical text features to extract keywords. It is independent of external dictionaries, corpora, languages, or domains, making it flexible and versatile.

## 2. Embedding-Based Methods

These methods use word embeddings (vector representations of words) to identify keywords. Examples include:

### 2.1 KeyBERT

KeyBERT leverages BERT embeddings and cosine similarity to identify keywords that are most similar to the overall document embedding. This method provides context-aware keyword extraction.

---

## 📚 Use Cases

- Search Engine Optimization (SEO)
- Information retrieval
- Text categorization
- Topic modeling

---


# 🚀 Getting Started

## 1. Install Miniconda in WSL
Run the following command to download the Miniconda installer:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

## 2. Create a new environment
Create a new environment with Python 3.9 using the following command:
```bash
conda create -n keywords python=3.9
```

## 3. Activate the environment
Activate the newly created environment:
```bash
conda activate keywords
```

## 4. Install requirements
Install the necessary dependencies using pip:
```bash
pip install -r requirements.txt
```

## 5. Run the project
To run the application, use the following command:
```bash
streamlit run app.py
```

Navigate to http://localhost:8501/ to access the application.

