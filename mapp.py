import streamlit as st
import pandas as pd
from fuzzywuzzy import fuzz
import spacy
from collections import Counter
import jellyfish
from glove_model import get_glove_model_result
from custom_model import get_custom_model_result
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from transformers import pipeline
import time
import pickle
import re
from sklearn import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Download required NLTK models if not downloaded
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt')
nltk.download('punkt_tab')

# Load English language model for Spacy
nlp = spacy.load('en_core_web_sm')

# Mapping of entity labels from NLTK to Standard Labels
entity_mapping = {
    'PERSON': 'PERSON',
    'ORG': 'ORGANIZATION',
    'ORGANIZATION': 'ORGANIZATION',
    'GPE': 'LOCATION',
    'LOC': 'LOCATION',
    'FAC': 'LOCATION',
    'FACILITY': 'LOCATION',
    'DATE': 'Miscellaneous',
    'TIME': 'Miscellaneous',
    'PERCENT': 'Miscellaneous',
    'MONEY': 'Miscellaneous',
    'QUANTITY': 'Miscellaneous',
    'ORDINAL': 'Miscellaneous',
    'EVENT': 'Miscellaneous',
    'WORK_OF_ART': 'Miscellaneous',
    'NORP': 'Miscellaneous',
    'CREATIVE_WORK': 'Miscellaneous'
}

# Scoring for entity labels
label_scoring = {
    'PERSON': 3,
    'ORGANIZATION': 2,
    'LOCATION': 1,
    'Miscellaneous': 0,
}

# List to store Spacy entities
spacy_entities = []

# Function to extract entities using NLTK
def extract_entity_nltk(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    chunked = nltk.ne_chunk(tagged_tokens)
    entity_labels = []
    for tree in chunked:
        if hasattr(tree, 'label'):
            entity_label = entity_mapping.get(tree.label(), 'Miscellaneous')
            entity_token = ''.join([token[0] for token in tree])
            spacy_entities.append(entity_token)
            entity_labels.append((entity_token, entity_label))
    return entity_labels

# Function to render Spacy NER
def render_spacy_ner(text):
    doc = nlp(text)
    entity_labels = []
    if doc.ents:
        for ent in doc.ents:
            entity_label = entity_mapping.get(ent.label_, 'Miscellaneous')
            entity_token = ent.text
            entity_labels.append((entity_token, entity_label))
    return entity_labels

# Function to render Transformers NER
def render_transformers_ner(text):
    pipe = pipeline("ner", model=".", tokenizer=".", aggregation_strategy="max")
    result = pipe(text)
    entity_labels = []
    for ent in result:
        entity_label = entity_mapping.get(ent['entity_group'], 'Miscellaneous')
        entity_token = ent['word']
        entity_confidence = ent['score']
        entity_labels.append((entity_token, entity_label, entity_confidence))
    return entity_labels

# Function to preprocess text
def preprocess_text(sen):
    sentence = re.sub('[^a-zA-Z]', '', sen)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", '', sentence)
    sentence = re.sub(r'\s+', '', sentence)
    return sentence

# Function to get final label based on scoring
def get_final_label(labels):
    label_frequency = Counter(labels)
    return max(label_frequency, key=lambda x: label_frequency[x] * label_scoring[x])

# Function to remove extra spaces
def remove_extra_spaces(s):
    return ''.join(s.split())

# Function to calculate fuzzy match score
def calculate_fuzzy_match_score(name1, name2):
    name1 = remove_extra_spaces(name1)
    name2 = remove_extra_spaces(name2)
    if not name1 or not name2 or not isinstance(name1, str) or not isinstance(name2, str):
        return 0
    return fuzz.token_sort_ratio(name1.lower(), name2.lower())

# Function to calculate similarity score
def calculate_similarity_score(name1, name2):
    doc1 = nlp(name1)
    doc2 = nlp(name2)
    similarity = doc1.similarity(doc2)
    return similarity * 100

# Function to calculate Levenshtein distance
def calculate_levenshtein_distance(name1, name2):
    max_len = max(len(name1), len(name2))
    distance = jellyfish.levenshtein_distance(name1, name2)
    score = (1 - (distance / max_len)) * 100
    return score

# Function to get similarity label
def get_similarity_label(similarity_score):
    if similarity_score > 90:
        return "Widely used synonyms for names"
    elif 75 <= similarity_score <= 90:
        return "Highly probable synonyms for names"
    elif 60 <= similarity_score <= 75:
        return "Possibly synonyms for names"
    else:
        return "Less likely synonyms for names"

# Function to calculate Jaro-Winkler distance
def calculate_jaro_winkler_distance(name1, name2):
    return jellyfish.jaro_winkler_similarity(name1, name2) * 100

# Function to calculate cosine similarity
def calculate_cosine_similarity(name1, name2):
    v1 = Counter(name1)
    v2 = Counter(name2)
    dot_product = sum(v1[key] * v2[key] for key in v1.keys() & v2.keys())
    magnitude1 = sum(v1[key]**2 for key in v1.keys()) ** 0.5
    magnitude2 = sum(v2[key]**2 for key in v2.keys()) ** 0.5
    return (dot_product / (magnitude1 * magnitude2)) * 100

# Function to process names
def process_names(names1, names2):
    data = []
    for name1, name2 in zip(names1, names2):
        fuzzy_score = calculate_fuzzy_match_score(name1, name2)
        similarity_score = calculate_similarity_score(name1, name2)
        levenshtein_score = calculate_levenshtein_distance(name1, name2)
        jw_score = calculate_jaro_winkler_distance(name1, name2)
        similarity_label = get_similarity_label(similarity_score)
        data.append({
            "Name 1": name1,
            "Name 2": name2,
            "Fuzzy Match Score": fuzzy_score,
            "Levenshtein Score": levenshtein_score,
            # "JW Score": jw_score,
            "AI Score": similarity_score,
            "Similarity": similarity_label
        })
    return pd.DataFrame(data, index=None)

# Function to convert DataFrame to CSV
def convert_df_to_csv(df):
    return df.to_csv(index=False)

# Page 1: Named Entity Similarity Recognition
def page1():
    st.title("Named Entity Similarity Recognition")
    
    col1, col2 = st.columns(2)
    
    name1_input = col1.text_input("Names 1 (comma-separated)", placeholder="John Smith, Jane Doe, Bob Johnson")
    name2_input = col2.text_input("Names 2 (comma-separated)", placeholder="J Smith, J Doe, B Johnson")
    
    if st.button("Calculate Scores"):
        with st.spinner('Calculating Scores...'):
            names1 = [name.strip() for name in name1_input.split(',')]
            names2 = [name.strip() for name in name2_input.split(',')]
            
            if len(names1) != len(names2):
                st.error("Please enter the same number of names in both boxes.")
            else:
                df = process_names(names1, names2)
                df.index += 1
                df.index.name = 'S No'
                
                st.write("Match Scores:")
                st.table(df)
                
                csv = convert_df_to_csv(df)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="fuzzy_match_scores.csv",
                    mime="text/csv",
                )

# Page 2: Ensemble AI - Intelligent Text Classification through Model Fusion
def page2():
    st.title("Ensemble AI: Intelligent Text Classification through Model Fusion")

    entities = st.text_input(
        "Enter entities separated by comma:", 
        placeholder="e.g., Credit Suisse, Google, London", 
        key="query"
    )

    csv_button = st.checkbox("Upload CSV here")
    csv_file = None
    csv_entities = []

    if csv_button:
        csv_file = st.file_uploader(
            "Please upload your CSV file", 
            type=["csv"]
        )

        if csv_file:
            csv_entities = [
                entity.split(",")[0] 
                for entity in csv_file.read().decode("utf-8").splitlines()[1:]
            ]

    if st.button("Run NER"):
        st.write("Generating response...")
        start_time = time.time()
        results = []
        entities_list = csv_entities if csv_button else [entity.strip() for entity in entities.split(',')]

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "NLTK NER", 
            "Spacy NER", 
            "Transformers NER", 
            "Glove Model NER", 
            "Custom Model NER"
        ])

        for entity in entities_list:
            global spacy_entities

            with tab1:
                nltk_labels = extract_entity_nltk(entity)
                for token, label in nltk_labels:
                    results.append({
                        'Token': token, 
                        'NLTK': label, 
                        'Spacy': '', 
                        'Transformers': '', 
                        'Glove Model': '', 
                        'Custom Model': ''
                    })
                    st.write(f"({token}) - ({label})")

            with tab2:
                spacy_labels = render_spacy_ner(entity)
                for token, label in spacy_labels:
                    existing_result = next((result for result in results if result['Token'] == token), None)
                    if existing_result:
                        existing_result['Spacy'] = label
                    else:
                        results.append({
                            'Token': token, 
                            'NLTK': "", 
                            'Spacy': label, 
                            'Transformers': '', 
                            'Glove Model': '', 
                            'Custom Model': ''
                        })
                    st.write(f"({token}) - ({label})")

            with tab3:
                transformers_labels = render_transformers_ner(entity)
                for token, label, confidence in transformers_labels:
                    existing_result = next((result for result in results if result['Token'] == token), None)
                    if existing_result:
                        existing_result['Transformers'] = label
                    else:
                        results.append({
                            'Token': token, 
                            'NLTK': "", 
                            'Spacy': '', 
                            'Transformers': label, 
                            'Glove Model': '', 
                            'Custom Model': ''
                        })
                    st.write(f"({token}) - ({label})")

            with tab4:
                for token in set(spacy_entities):
                    label, confidence = get_glove_model_result(token)
                    existing_result = next((result for result in results if result['Token'] == token), None)
                    if existing_result:
                        existing_result['Glove Model'] = label
                    else:
                        results.append({
                            'Token': token, 
                            'NLTK': '', 
                            'Spacy': '', 
                            'Transformers': '', 
                            'Glove Model': label, 
                            'Custom Model': ''
                        })
                    st.write(f"({token}) - ({label})")

            with tab5:
                for token in set(spacy_entities):
                    label, confidence = get_custom_model_result(token)
                    existing_result = next((result for result in results if result['Token'] == token), None)
                    if existing_result:
                        existing_result['Custom Model'] = label
                    else:
                        results.append({
                            'Token': token, 
                            'NLTK': '', 
                            'Spacy': '', 
                            'Transformers': '', 
                            'Glove Model': '', 
                            'Custom Model': label
                        })
                    st.write(f"({token}) - ({label})")
            spacy_entities = []

        for result in results:
            labels = [label for label in [result['NLTK'], result['Spacy'], result['Transformers'], result['Glove Model'], result['Custom Model']] if label]
            result['Final Label'] = get_final_label(labels)

        df = pd.DataFrame(results)
        st.markdown("## Summary:")
        st.write(df)


        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(df)
        st.download_button(
            label="Download as CSV", 
            data=csv, 
            file_name="Ensemble results.csv", 
            mime='text/csv'
        )

        end_time = time.time()
        total_time_elapsed = end_time - start_time
        st.write(f"Total Time Elapsed: {total_time_elapsed:.2f} seconds")

# Create a sidebar with radio buttons to choose the page
page = st.sidebar.radio(
    "Choose your application", 
    ["Similarity", "NER"]
)

# Display the correct page based on the radio button choice
page1_container = st.container()
page2_container = st.container()

if page == "Similarity":
    with page1_container:
        page1()
elif page == "NER":
    with page2_container:
        page2()