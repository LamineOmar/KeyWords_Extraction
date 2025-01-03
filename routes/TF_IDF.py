import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK resources
nltk.download('punkt')  # For word_tokenize and sent_tokenize
nltk.download('wordnet')  # For lemmatization
nltk.download('omw-1.4')  # Optional: For wordnet lemmatizer support

# Load the English model
nlp = spacy.load("en_core_web_sm")

class TFIDFProcessor:

    def lemmatization(self, text):

        doc = nlp(text)
        # Lemmatize
        lemmatized_words = [token.lemma_ for token in doc if not token.is_punct]
        return " ".join(lemmatized_words)
    
    
    def TF_IDF(self, text, num_keywords):

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_features=num_keywords, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform([text])

        # Extract keywords
        terms = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray().flatten()
        keywords = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)[:num_keywords]
        return keywords
