import streamlit as st
import pandas as pd
from rake_nltk import Rake
from keybert import KeyBERT
from yake import KeywordExtractor as YAKE
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import re
from routes.TF_IDF import TFIDFProcessor

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# Set up Streamlit page
st.set_page_config(page_title="Keyword Extractor", page_icon="🔑")
st.title("🔑 Keyword Extractor")

# Upload document
uploaded_file = st.file_uploader("Upload a text document", type="txt")
if uploaded_file is not None:
    doc = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
    st.write("Document loaded successfully!")
else:
    st.stop()

method = st.radio("Choose extraction method", ("TF-IDF", "RAKE", "YAKE", "KeyBERT"))

num_keywords = st.slider("Number of Keywords", 2, 10, 5)

# Extract keywords
keywords = []
doc = re.sub(r"[^a-zA-Z0-9\s]", "", doc)


if method == "TF-IDF":
    tf_idf =TFIDFProcessor()
    lemmatized = tf_idf.lemmatization(doc)
    keywords = tf_idf.TF_IDF(lemmatized,num_keywords)

elif method == "RAKE":
    rake = Rake()
    rake.extract_keywords_from_text(doc)
    keywords = rake.get_ranked_phrases_with_scores()[:num_keywords]

elif method == "KeyBERT":
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), top_n=num_keywords)
    
elif method == "YAKE":
    yake_extractor = YAKE()
    keywords = yake_extractor.extract_keywords(doc)
    keywords_sorted = sorted(keywords, key=lambda x: x[1], reverse=True)  
    keywords = keywords_sorted[:num_keywords]
# Display keywords
st.write("### Extracted Keywords")
if method == "RAKE":
    st.table(pd.DataFrame(keywords, columns=["Score", "Keyword"]))
    KeyW = [kw for score,kw in keywords]
else:
    st.table(pd.DataFrame(keywords, columns=["Keyword", "Relevancy"]))
    KeyW = [kw for kw, score in keywords]


# Visualization

if method != "RAKE":
    st.write("### Keyword Relevancy Visualization")
    df = pd.DataFrame(keywords, columns=["Keyword", "Relevancy"])
    df["Relevancy"] = df["Relevancy"].astype(float)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Relevancy", y="Keyword", data=df)
    plt.title("Keyword Relevancy")
    st.pyplot(plt)

# Download extracted keywords
df = pd.DataFrame(keywords, columns=["Score" ,"Keyword"] if method == "RAKE" else ["Keyword", "Relevancy"])
csv = df.to_csv(index=False)
st.download_button("Download CSV", csv, "keywords.csv", "text/csv")



# Fonction pour surligner les mots-clés dans le texte
def highlight_keywords(text, keywords):
    for i, keyword in enumerate(keywords):
        color = f"hsl({i * 50 % 360}, 70%, 70%)"  # Générer une couleur différente pour chaque mot-clé
        # Utiliser une expression régulière pour une recherche insensible à la casse
        keyword = str(keyword)
        regex = re.compile(re.escape(keyword), re.IGNORECASE)
        # Remplacer les occurrences du mot-clé par une version colorée
        text = regex.sub(f"<span style='color: {color}; font-weight: bold;'>{keyword}</span>", text)
    return text
print(KeyW)
# Appliquer le surlignage des mots-clés
highlighted_text = highlight_keywords(doc, KeyW)

# Afficher le texte formaté dans Streamlit
st.markdown(f"<p style='font-size:18px;'>{highlighted_text}</p>", unsafe_allow_html=True)
