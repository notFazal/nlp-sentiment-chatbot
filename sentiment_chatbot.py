from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from nltk.parse.corenlp import CoreNLPDependencyParser
import pandas as pd
import numpy as np
import pickle as pkl
import string
import re
import csv
import nltk
from time import localtime, strftime

f = open("{0}.txt".format(strftime("%Y-%m-%d_%H-%M-%S", localtime())), "w")

EMBEDDING_FILE = "glove.pkl"


def load_glove(filepath):
    with open(filepath, 'rb') as fin:
        return pkl.load(fin)


def load_as_list(fname):
    df = pd.read_csv(fname)
    documents = df['review'].values.tolist()
    labels = df['label'].values.tolist()
    return documents, labels


def extract_user_info(user_input):
    name = ""
    name_match = re.search(r"(^|\s)([A-Z][A-Za-z-&'\.]*(\s|$)){2,4}", user_input)
    if name_match is not None:
        name = name_match.group(0).strip()
    return name


def get_tokens(inp_str):
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    return nltk.tokenize.word_tokenize(inp_str)


def vectorize_train(training_documents):
    vectorizer = TfidfVectorizer(tokenizer=get_tokens, lowercase=True)
    tfidf_train = vectorizer.fit_transform(training_documents)
    return vectorizer, tfidf_train


def glove(glove_reps, token):
    word_vector = np.zeros(300,)
    if token in glove_reps:
        word_vector = glove_reps[token]
    return word_vector


def string2vec(glove_reps, user_input):
    tokens = get_tokens(user_input)
    if not tokens:
        return np.zeros(300,)
    vectors = [glove(glove_reps, t) for t in tokens]
    return np.mean(vectors, axis=0)


def instantiate_models():
    nb = GaussianNB()
    logistic = LogisticRegression(random_state=100)
    svm = LinearSVC(random_state=100)
    mlp = MLPClassifier(random_state=100)
    return nb, logistic, svm, mlp


def train_model_tfidf(model, tfidf_train, training_labels):
    X = tfidf_train.toarray()
    model.fit(X, training_labels)
    return model


def train_model_glove(model, glove_reps, training_documents, training_labels):
    doc_vectors = [string2vec(glove_reps, doc) for doc in training_documents]
    X_train = np.vstack(doc_vectors)
    model.fit(X_train, training_labels)
    return model


def test_model_tfidf(model, vectorizer, test_documents, test_labels):
    X_test = vectorizer.transform(test_documents).toarray()
    preds = model.predict(X_test)
    precision = precision_score(test_labels, preds)
    recall = recall_score(test_labels, preds)
    f1 = f1_score(test_labels, preds)
    accuracy = accuracy_score(test_labels, preds)
    return precision, recall, f1, accuracy


def test_model_glove(model, glove_reps, test_documents, test_labels):
    X_test = np.array([string2vec(glove_reps, doc) for doc in test_documents])
    preds = model.predict(X_test)
    precision = precision_score(test_labels, preds)
    recall = recall_score(test_labels, preds)
    f1 = f1_score(test_labels, preds)
    accuracy = accuracy_score(test_labels, preds)
    return precision, recall, f1, accuracy


def get_dependency_parse(input: str):
    output = ""
    dep_parser = CoreNLPDependencyParser(url="http://localhost:9000")
    try:
        parses = dep_parser.raw_parse(input)
        parse = next(parses)
    except Exception:
        return ""

    nodes = parse.nodes
    for addr in sorted(nodes.keys()):
        if addr == 0:
            continue
        node = nodes[addr]
        word = node.get("word", "")
        pos = node.get("tag", "")
        head = node.get("head", 0)
        rel = node.get("rel", "")
        output += f"{word}\t{pos}\t{head}\t{rel}\n"

    return output


def get_dep_categories(parsed_input):
    num_nsubj = 0
    num_obj = 0
    num_iobj = 0
    num_nmod = 0
    num_amod = 0

    if not parsed_input:
        return 0, 0, 0, 0, 0

    for line in parsed_input.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        relation = parts[3]
        if relation.startswith("nsubj"):
            num_nsubj += 1
        elif relation in ("obj", "dobj") or relation.startswith("obj:") or relation.startswith("dobj"):
            num_obj += 1
        elif relation.startswith("iobj"):
            num_iobj += 1
        elif relation.startswith("nmod"):
            num_nmod += 1
        elif relation.startswith("amod"):
            num_amod += 1

    return num_nsubj, num_obj, num_iobj, num_nmod, num_amod


def custom_feature(user_input):
    sentences = nltk.sent_tokenize(user_input)
    if not sentences:
        return "Average sentence length: 0"
    lengths = [len(get_tokens(s)) for s in sentences]
    avg_len = sum(lengths) / len(lengths)
    return f"Average sentence length: {avg_len:.2f}"


def welcome_state():
    print("Welcome to the NLP Sentiment Chatbot!")
    f.write("CHATBOT:\nWelcome to the NLP Sentiment Chatbot!\n")
    return "get_user_info"


def get_info_state():
    user_input = input("What is your name?\n")
    f.write("What is your name?\n")
    f.write("\nUSER:\n{0}\n".format(user_input))
    name = extract_user_info(user_input)
    return "sentiment_analysis", name


def sentiment_analysis_state(name, model, vectorizer=None, glove_reps=None):
    user_input = input("Thanks {0}! What do you want to talk about today?\n".format(name))
    f.write("\nCHATBOT:\nThanks {0}! What do you want to talk about today?\n".format(name))
    f.write("\nUSER:\n{0}\n".format(user_input))

    test = string2vec(glove_reps, user_input)
    label = model.predict(test.reshape(1, -1))

    if label == 0:
        print("Hmm, it seems like you're feeling a bit down.")
        f.write("\nCHATBOT:\nHmm, it seems like you're feeling a bit down.\n")
    elif label == 1:
        print("It sounds like you're in a positive mood!")
        f.write("\nCHATBOT:\nIt sounds like you're in a positive mood!\n")
    else:
        print("Hmm, that's weird. My classifier predicted a value of: {0}".format(label))
        f.write("\nCHATBOT:\nHmm, that's weird. My classifier predicted a value of: {0}\n".format(label))

    return "stylistic_analysis"


def stylistic_analysis_state():
    user_input = input("I'd also like to do a quick stylistic analysis. What's on your mind today?\n")
    f.write("\nCHATBOT:\nI'd also like to do a quick stylistic analysis. What's on your mind today?\n")
    f.write("\nUSER:\n{0}\n".format(user_input))

    dep_parse = get_dependency_parse(user_input)
    num_nsubj, num_obj, num_iobj, num_nmod, num_amod = get_dep_categories(dep_parse)
    custom = custom_feature(user_input)

    print("Thanks! Here's what I discovered about your writing style.")
    print("# Nominal Subjects: {0}\n# Direct Objects: {1}\n# Indirect Objects: {2}"
          "\n# Nominal Modifiers: {3}\n# Adjectival Modifiers: {4}".format(
              num_nsubj, num_obj, num_iobj, num_nmod, num_amod))
    print("Custom Feature: {0}".format(custom))

    f.write("\nCHATBOT:\nThanks! Here's what I discovered about your writing style.\n")
    f.write("# Nominal Subjects: {0}\n# Direct Objects: {1}\n# Indirect Objects: {2}"
          "\n# Nominal Modifiers: {3}\n# Adjectival Modifiers: {4}\n".format(
              num_nsubj, num_obj, num_iobj, num_nmod, num_amod))
    f.write("Custom Feature: {0}\n".format(custom))

    return "check_next_action"


def check_next_state():
    user_input = input("What would you like to do next? You can quit, redo the "
                       "sentiment analysis, or redo the stylistic analysis.\n")
    f.write("\nCHATBOT:\nWhat would you like to do next? You can quit, redo the "
                       "sentiment analysis, or redo the stylistic analysis.\n")
    f.write("\nUSER:\n{0}\n".format(user_input))

    match = False
    while not match:
        if re.search(r"\bquit\b", user_input):
            return "quit"
        elif re.search(r"\bsentiment\b", user_input):
            return "sentiment_analysis"
        elif re.search(r"\bstyl", user_input):
            return "stylistic_analysis"
        else:
            user_input = input("Sorry, I didn't understand that. Would you like "
                               "to quit, redo the sentiment analysis, or redo the stylistic analysis?\n")
            f.write("\nCHATBOT:\nSorry, I didn't understand that.\n")
            f.write("\nUSER:\n{0}\n".format(user_input))


def run_chatbot(model, vectorizer=None, glove_reps=None):
    next_state = welcome_state()
    sentiment_analysis_counter = 0

    while next_state != "quit":
        if next_state == "get_user_info":
            next_state, name = get_info_state()
        elif next_state == "sentiment_analysis":
            next_state = sentiment_analysis_state(name, model, vectorizer, glove_reps)
            if sentiment_analysis_counter > 0:
                next_state = "check_next_action"
            sentiment_analysis_counter += 1
        elif next_state == "stylistic_analysis":
            next_state = stylistic_analysis_state()
        elif next_state == "check_next_action":
            next_state = check_next_state()

    f.close()


if __name__ == "__main__":
    documents, labels = load_as_list("dataset.csv")
    glove_reps = load_glove(EMBEDDING_FILE)

    nb_glove, logistic_glove, svm_glove, mlp_glove = instantiate_models()
    svm_glove = train_model_glove(svm_glove, glove_reps, documents, labels)

    run_chatbot(svm_glove, glove_reps=glove_reps)
    f.close()
