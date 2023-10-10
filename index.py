import string
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
from collections import Counter
from math import log10 as log
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from time import time
from pickle import load, dump
from os import listdir, getcwd
from string import punctuation
from math import sqrt
from itertools import chain
import gc


def clean_my_depression(variable):
    del variable
    gc.collect()


def clean_text(specified_text):
    cleaned_string_table = specified_text.maketrans("", "", punctuation)
    cleaned_string = specified_text.translate(cleaned_string_table)
    return cleaned_string


def clean_stop_words(stop_words, tokens):
    return (token for token_list in tokens for token in token_list if token.lower() not in stop_words)


def lemmatize_terms(lemmatizer, tokens):
    return [lemmatizer.lemmatize(term) for token_list in tokens for term in token_list]


def stemmerize_terms(porter_stemmer, tokens):

    return [porter_stemmer.stem(term) for token_list in tokens for term in token_list]


def retrieve_directory_files(specified_directory=getcwd()):
    return listdir(specified_directory)


def soupify_document(specified_document):
    with open("./ueapeople/" + specified_document, encoding="utf8") as file:
        parsed_document = BeautifulSoup(file.read(), "lxml")
        file.close()
    return parsed_document


def retrieve_document_paragraphs(specified_document):
    soupified_document = soupify_document(specified_document)
    retrieved_paragraphs = [paragraph.text for paragraph in soupified_document.select("p")]
    cleaned_paragraphs = [clean_text(paragraph) for paragraph in retrieved_paragraphs]
    return cleaned_paragraphs


def construct_token_counter(tokens):
    constructed_list = (token for token_list in tokens for token in token_list)
    return Counter(constructed_list)


def serialize_unique_terms():
    placeholder_term_list = []
    with open("serialized_unique_terms.txt", "ab") as fh:
        for document in retrieve_directory_files("./ueapeople"):
            paragraphs = retrieve_document_paragraphs(document)
            terms = [word_tokenize(sentence) for sentence in paragraphs]
            deconstructed_terms = list(chain(*terms))
            for term in deconstructed_terms:
                if term not in placeholder_term_list:
                    placeholder_term_list.append(term)
            print(f"Scanned document: {document}")
        dump(placeholder_term_list, fh)


def deserialize_unique_terms():
    placeholder_list = []
    with open("serialized_unique_terms.txt", "rb") as unserialized_pickle_data:
        try:
            while True:
                deserialized_data = load(unserialized_pickle_data)
                placeholder_list.append(deserialized_data)
        except EOFError:
            pass
    return placeholder_list


def serialize_paragraphs():
    placeholder_paragraph_list = []
    with open("serialized_paragraphs.txt", "ab") as file:
        for document in retrieve_directory_files("./ueapeople"):
            paragraphs = retrieve_document_paragraphs(document)
            placeholder_paragraph_list.append(paragraphs)
            print(f"Scanned document paragraphs: {document}")
        dump(placeholder_paragraph_list, file)


def deserialize_paragraphs():
    placeholder_paragraph_list = []
    with open("serialized_paragraphs.txt", "rb") as pickle_data:
        try:
            while True:
                deserialized_data = load(pickle_data)
                placeholder_paragraph_list.append(deserialized_data)
        except EOFError:
            pass
    return placeholder_paragraph_list


def build_incidence_matrix_skeleton():
    placeholder_incidence_matrix = {}
    for document in retrieve_directory_files("./ueapeople"):
        paragraphs = retrieve_document_paragraphs(document)
        terms = [word_tokenize(sentence) for sentence in paragraphs]
        deconstructed_terms = list(chain(*terms))
        for term in deconstructed_terms:
            if term not in placeholder_incidence_matrix:
                placeholder_incidence_matrix[term] = []
        print(f"Scanned document: {document}")
    return placeholder_incidence_matrix



def build_incidence_matrix(documents, unique_term_list):
    incidence_matrix = {}

    for u_term in unique_term_list:
        incidence_matrix[u_term] = []

    for document in documents:
        paragraphs = retrieve_document_paragraphs(document)
        terms = (word_tokenize(sentence) for sentence in paragraphs)
        deconstructed_terms = set(chain(*terms))
        for term in unique_term_list:
            incidence_matrix[term].append(int(term in deconstructed_terms))

    return incidence_matrix


def display_graph(test_name, document_names, document_data):
    fig = plt.figure(figsize=(10, 7))
    plt.bar(document_names, document_data, color='maroon', width=0.5)
    plt.xlabel("Documents")
    plt.ylabel("Document Weighting")
    plt.title(test_name)
    plt.setp(fig.get_axes()[0].get_xticklabels(), rotation=40)
    plt.setp(fig.get_axes()[0].get_yticklabels(), rotation=40)
    plt.savefig(f"{test_name}.png")
    plt.show()


def calculate_tf_idf(term_frequency, document_frequency, documents_size):
    calculated_tf = log(1 + term_frequency)
    if document_frequency == 0:
        calculated_idf = 0
    else:
        calculated_idf = log(documents_size / document_frequency)
    calculated_tf_idf = calculated_tf * calculated_idf
    return calculated_tf_idf if term_frequency != 0 else 0


def calculate_tf_idf_list(frequency_vector, unique_term_list, document_frequency, documents_size):
    return [calculate_tf_idf(term_frequency, document_frequency[term], documents_size) for term, term_frequency in
            zip(unique_term_list, frequency_vector)]


def l2_normalize_vector(specified_vector):
    placeholder_vector = []
    length_normalised_vector_value = sqrt(sum(v ** 2 for v in specified_vector))
    for v in specified_vector:
        if length_normalised_vector_value == 0:
            placeholder_vector.append(0)
        else :
            placeholder_vector.append(v / length_normalised_vector_value)
    return placeholder_vector


def calculate_query_tdidf(specified_query, unique_term_list, document_frequency, documents_size):
    deconstructed_query = word_tokenize(specified_query)
    query_counter = Counter(deconstructed_query)
    vectorised_term_frequency = [query_counter[term] for term in unique_term_list]
    print(f"Vectorised Term Frequency: {vectorised_term_frequency}")
    calculated_query_tfidf = calculate_tf_idf_list(vectorised_term_frequency, unique_term_list, document_frequency,
                                                   documents_size)
    return calculated_query_tfidf


def dot_product(first_vector, second_vector):
    return sum(v1 * v2 for v1, v2 in zip(first_vector, second_vector))



def query(specified_query, tfidf_vectors, unique_term_list, document_frequency, documents_size, documents_list):
    print(f"Specified Query: {specified_query}")
    query_tfidf = calculate_query_tdidf(specified_query, unique_term_list, document_frequency, documents_size)
    print(f"Query tf-idf: {query_tfidf}")
    normalized_query_tfidf_vectors = l2_normalize_vector(query_tfidf)
    normalized_tfidf_vectors = (l2_normalize_vector(vector) for vector in tfidf_vectors)
    dot_products = [dot_product(normalized_query_tfidf_vectors, vector) for vector in normalized_tfidf_vectors]
    print(f"Dot Products: {dot_products}")
    sorted_dot_products = sorted(dot_products, key=lambda x: float(x))
    print(f"Sorted Dot Products: {sorted_dot_products}")
    top_ten_dot_products = sorted_dot_products[-10:]
    print(f"Top Ten Dot Products: {top_ten_dot_products[::-1]}")
    closest_document_match = max(dot_products)
    print(f"Closest Document Match: {closest_document_match}")
    calculated_indexes = (i for i, j in enumerate(dot_products) if j == closest_document_match)
    calculated_top_ten_indexes = list((dot_products.index(d_product) for d_product in top_ten_dot_products))
    top_ten_document_names = [f"{i} - {documents_list[calculated_index]}" for i, calculated_index in enumerate(calculated_top_ten_indexes)]
    print(f"Top Ten Document Names: {top_ten_document_names}")
    display_graph(f"Stemmer - {specified_query}", top_ten_document_names, top_ten_dot_products)
    return list((documents_list[calculated_index] for calculated_index in calculated_indexes))


def main():
    execution_start_time = time()
    unique_term_list = []
    document_list = []
    counters = Counter()
    document_counters = []
    documents_size = 5000
    stop_words = stopwords.words("english")
    porter_stemmer = PorterStemmer()
    # lemmatizer = WordNetLemmatizer()

    for i in range(documents_size):
        file_name = f"page{i}.html"
        document_list.append(file_name)
        paragraphs = retrieve_document_paragraphs(f"page{i}.html")
        tokens = (word_tokenize(sentence) for sentence in paragraphs)
        #tokens = clean_stop_words(stop_words, tokens)
        #tokens = lemmatize_terms(lemmatizer, tokens)
        tokens = stemmerize_terms(porter_stemmer, tokens)
        token_counters = Counter(tokens)
        document_counters.append(token_counters)
        counters.update(token_counters)
        for key in list(counters.keys()):
            unique_term_list.append(key)

    # Test for looping every paragraph as well.
    # paragraphs = deserialize_paragraphs()
    # document_paragraphs = list(chain(*paragraphs))
    # document_list = []
    # for deconstructed_paragraphs in paragraphs:
    #     for sentence in deconstructed_paragraphs:
    #         tokens = (word_tokenize(s) for s in sentence)
    #         tokens = clean_stop_words(stop_words, tokens)
    #         token_counters = Counter(tokens)
    #         document_counters.append(token_counters)
    #         counters.update(token_counters)
    #         for key in list(counters.keys()):
    #             unique_term_list.append(key)
    #
    # del paragraphs
    # del document_paragraphs
    # gc.collect()
    # del counters
    # gc.collect()
    #
    # documents_size = 17909
    # for i in range(documents_size):
    #     file_name = f"page{i}.html"
    #     document_list.append(file_name)

    unique_term_list = list(set(unique_term_list))
    print("Unique term list built!")
    incidence_matrix = build_incidence_matrix(document_list, unique_term_list)
    print("Incidence matrix built!")
    document_frequency_matrix = {k: sum(v) for k, v in incidence_matrix.items()}
    print("Document Frequency Matrix built!")
    del incidence_matrix
    gc.collect()
    vectors = ((document_count[term] for term in unique_term_list) for document_count in document_counters)
    print("Vectors built!")
    del document_counters
    gc.collect()
    tf_idf_list = (calculate_tf_idf_list(frequency_vector, unique_term_list, document_frequency_matrix, documents_size)
                   for frequency_vector in vectors)
    print("TF-IDF Vector built!")
    del vectors
    gc.collect()
    print(query("researchers from other universities and with official agencies", tf_idf_list, unique_term_list, document_frequency_matrix, documents_size,
                document_list))


    execution_finish_time = time()
    print(f"Code took: {(execution_finish_time - execution_start_time)} seconds!")


if __name__ == '__main__':
    main()
