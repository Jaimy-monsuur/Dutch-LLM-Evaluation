import csv
import os

import numpy as np
import pdfplumber  # For extracting text from PDF
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from bert_score import score, BERTScorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch
from tqdm import tqdm  # For progress bar
from nltk.corpus import stopwords
import nltk
# set the nltk to Dutch

from evaluate import load
# Metrics for evaluation
def calculate_perplexity(text):
    model_name = "GroNLP/gpt2-small-dutch"
    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(predictions=text, model_id=model_name)
    mean_perplexity = np.mean(results["perplexities"])
    return mean_perplexity


def calculate_similarity(original_text, summary_text):
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode([original_text, summary_text])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity



# Download Dutch stop words
nltk.download('stopwords')
dutch_stopwords = stopwords.words('dutch')


def remove_stopwords(text, stop_words):
    """Remove stop words from text."""
    return " ".join([word for word in text.split() if word.lower() not in stop_words])


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
import csv


def calculate_keywords_coverage(original_text, summary_text, tfidf_threshold=0.1):
    """
    Calculate keyword coverage by comparing keywords from the original text to the summary.

    Args:
        original_text (str): The original text.
        summary_text (str): The summary text.
        tfidf_threshold (float): Minimum TF-IDF score to consider a word as a keyword.

    Returns:
        float: Keyword coverage as the fraction of original keywords found in the summary.
    """
    # Remove stop words and tokenize text
    original_text_clean = remove_stopwords(original_text, dutch_stopwords)
    summary_text_clean = remove_stopwords(summary_text, dutch_stopwords)

    # Tokenize for better handling of punctuation and multi-word tokens
    original_tokens = word_tokenize(original_text_clean)
    summary_tokens = word_tokenize(summary_text_clean)

    # Vectorize to extract keywords and their weights from the original text
    vectorizer = TfidfVectorizer(stop_words=dutch_stopwords)
    tfidf_matrix = vectorizer.fit_transform([original_text_clean])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]

    # Extract keywords with TF-IDF scores above the threshold
    original_keywords = {
        feature_names[i] for i, score in enumerate(tfidf_scores) if score > tfidf_threshold
    }
    summary_keywords = set(summary_tokens)

    # Compute overlap
    overlapping_keywords = original_keywords.intersection(summary_keywords)

    print("Original Keywords:", original_keywords)
    print("Summary Keywords:", summary_keywords)
    print("Overlapping Keywords:", overlapping_keywords)

    # Avoid division by zero
    if len(original_keywords) == 0:
        return 0.0

    coverage = len(overlapping_keywords) / len(original_keywords)

    # Save results to a separate CSV
    with open("keywords_coverage_details.csv", mode="a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Original Keywords", "Summary Keywords", "Overlapping Keywords"
        ])
        writer.writerow([
            ", ".join(original_keywords),
            ", ".join(summary_tokens),
            ", ".join(overlapping_keywords)
        ])

    return coverage


from transformers import pipeline

# Load a NER pipeline for Dutch (you can use a model from Hugging Face)
ner_pipeline = pipeline("ner", model="wietsedv/bert-base-dutch-cased-finetuned-conll2002-ner", aggregation_strategy="simple")

def extract_entities(text):
    """Extract unique named entities from the text."""
    entities = ner_pipeline(text)
    # Get unique entity texts
    return set(entity["word"] for entity in entities)

def calculate_entity_coverage(original_text, summary_text):
    """Calculate entity coverage as the fraction of entities in the summary that are also in the original text."""
    original_entities = extract_entities(original_text)
    summary_entities = extract_entities(summary_text)

    # Compute overlap
    overlapping_entities = original_entities.intersection(summary_entities)
    print("Overlapping entities:")
    print(overlapping_entities)

    # Avoid division by zero
    if len(original_entities) == 0:
        return 0.0

    # Save results to a separate CSV
    with open("entities_coverage_details.csv", mode="a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Original Entities", "Summary Entities", "Overlapping Entities"
        ])
        writer.writerow([
            ", ".join(original_entities),
            ", ".join(summary_entities),
            ", ".join(overlapping_entities)
        ])

    coverage = len(overlapping_entities) / len(original_entities)
    return coverage


from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

def calculate_topic_coverage(original_text, summary_text, n_topics=6, topic_threshold=0.1):
    """
    Calculate topic coverage using Latent Dirichlet Allocation (LDA) and output topic words.

    Args:
        original_text (str): The original text.
        summary_text (str): The summary text.
        n_topics (int): Number of topics for LDA.
        topic_threshold (float): Minimum weight for a topic to be considered dominant.

    Returns:
        float: Topic coverage as the fraction of topics in the original text that are also present in the summary.
    """
    # Vectorize the texts
    vectorizer = CountVectorizer(stop_words=dutch_stopwords)
    vectorized_texts = vectorizer.fit_transform([original_text, summary_text])

    # Fit LDA on the combined vectorized text
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(vectorized_texts)

    # Extract top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    topic_words = {
        topic_idx: [feature_names[i] for i in topic.argsort()[:-6:-1]]
        for topic_idx, topic in enumerate(lda.components_)
    }

    # Get topic distributions for original and summary
    topic_distributions = lda.transform(vectorized_texts)
    original_topic_distribution = topic_distributions[0]
    summary_topic_distribution = topic_distributions[1]

    # Determine dominant topics based on threshold
    original_dominant_topics = {
        idx for idx, weight in enumerate(original_topic_distribution) if weight > topic_threshold
    }
    summary_dominant_topics = {
        idx for idx, weight in enumerate(summary_topic_distribution) if weight > topic_threshold
    }

    # Map dominant topics to words
    original_topic_words = {
        word for idx in original_dominant_topics for word in topic_words[idx]
    }
    summary_topic_words = {
        word for idx in summary_dominant_topics for word in topic_words[idx]
    }

    # Calculate overlapping words
    overlapping_topic_words = original_topic_words.intersection(summary_topic_words)

    print("Original dominant topic words:", original_topic_words)
    print("Summary dominant topic words:", summary_topic_words)
    print("Overlapping topic words:", overlapping_topic_words)

    # Avoid division by zero
    if len(original_topic_words) == 0:
        return 0.0

    coverage = len(overlapping_topic_words) / len(original_topic_words)

    # Save details to a CSV
    with open("topic_coverage_details.csv", mode="a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Original Topics", "Summary Topics", "Overlapping Topics"
        ])
        writer.writerow([
            ", ".join(original_topic_words),
            ", ".join(summary_topic_words),
            ", ".join(overlapping_topic_words)
        ])

    return coverage



def calculate_bertscore(original_text, summary_text):
    P, R, F1 = score([summary_text], [original_text], model_type="bert-base-multilingual-cased", lang="nl", verbose=True) #bert-base-multilingual-cased
    return P.mean().item(), R.mean().item(), F1.mean().item()

def calculate_reduction_factor(original_text, summary_text):
    """Calculate the reduction factor as the ratio of summary length to original text length."""
    original_length = len(original_text.split())
    summary_length = len(summary_text.split())
    if original_length == 0:
        return 0.0  # Avoid division by zero
    return summary_length / original_length

# pip install git+https://github.com/PrimerAI/blanc.git
# pip install --no-deps git+https://github.com/PrimerAI/blanc.git
from blanc import BlancHelp
def calculate_blanc(document, summary):
    print ("Calculating BLANC score")
    # GroNLP/bert-base-dutch-cased
    #bert-base-multilingual-cased
    blanc_help = BlancHelp( model_name="GroNLP/bert-base-dutch-cased")
    result = blanc_help.eval_once(document, summary)
    return result


# Folder path
folder_path = "C:\\Users\\jaimy\\Desktop\\overheids-brieven\\Brieven\\Ministerie-van-Onderwijs-Cultuuren"

# Separate letters and summaries
files = os.listdir(folder_path)
letters = [f for f in files if f.endswith(".pdf")]
summaries = [f for f in files if f.endswith(".txt")]

# Prepare results list
results = []

# Iterate over the files with progress tracking
for letter_file in tqdm(letters, desc="Processing letters", unit="letter"):
    letter_base = letter_file.replace(".pdf", "")  # Get the base name
    letter_path = os.path.join(folder_path, letter_file)

    # Extract text from the letter PDF
    with pdfplumber.open(letter_path) as pdf:
        original_text = " ".join(page.extract_text() for page in pdf.pages)

    # Find corresponding summaries
    related_summaries = [s for s in summaries if s.startswith(letter_base)]

    for summary_file in related_summaries:
        summary_path = os.path.join(folder_path, summary_file)
        print(f"Evaluating {letter_file} with {summary_file}")
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_text = f.read()

            # Apply metrics
            perplexity = calculate_perplexity(summary_text)
            similarity = calculate_similarity(original_text, summary_text)
            keyword_coverage = calculate_keywords_coverage(original_text, summary_text)
            entity_coverage = calculate_entity_coverage(original_text, summary_text)
            topic_coverage = calculate_topic_coverage(original_text, summary_text)
            # precision, recall, f1 = calculate_bertscore(original_text, summary_text)
            blanc_score = calculate_blanc(original_text, summary_text)
            reduction_factor = calculate_reduction_factor(original_text, summary_text)

            # Store results
            results.append({
                "letter": letter_file,
                "summary": summary_file,
                "perplexity": perplexity,
                "similarity": similarity,
                "keyword_coverage": keyword_coverage,
                "entity_coverage": entity_coverage,
                "topic_coverage": topic_coverage,
                # "bert_precision": precision,
                # "bert_recall": recall,
                # "bert_f1": f1,
                "blanc_score": blanc_score,
                "reduction_factor": reduction_factor
            })

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv("evaluation_results_dutch.csv", index=False)
