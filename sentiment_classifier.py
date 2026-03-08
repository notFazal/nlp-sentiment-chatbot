import pandas as pd
import numpy as np
import pickle as pkl
import nltk
import time
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

nltk.download('punkt_tab')

EMBEDDING_FILE = "glove.pkl"


def load_glove(filepath):
    with open(filepath, 'rb') as fin:
        return pkl.load(fin)


def load_as_list(fname):
    df = pd.read_csv(fname)
    documents = df['review'].values.tolist()
    labels = df['label'].values.tolist()
    return documents, labels


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


if __name__ == "__main__":
    print("Loading dataset...")
    documents, labels = load_as_list("dataset.csv")

    print("Loading GloVe representations...")
    glove_reps = load_glove(EMBEDDING_FILE)

    print("Computing TF-IDF representations...")
    vectorizer, tfidf_train = vectorize_train(documents)

    print("Instantiating models...")
    nb_tfidf, logistic_tfidf, svm_tfidf, mlp_tfidf = instantiate_models()
    nb_glove, logistic_glove, svm_glove, mlp_glove = instantiate_models()

    model_names = ["Naive Bayes", "Logistic Regression", "SVM", "Multilayer Perceptron"]
    models_tfidf_list = [nb_tfidf, logistic_tfidf, svm_tfidf, mlp_tfidf]
    models_glove_list = [nb_glove, logistic_glove, svm_glove, mlp_glove]
    train_funcs = [
        lambda m: train_model_tfidf(m, tfidf_train, labels),
        lambda m: train_model_glove(m, glove_reps, documents, labels),
    ]

    print("\nTraining models...")
    trained_tfidf = []
    trained_glove = []
    for name, mt, mg in zip(model_names, models_tfidf_list, models_glove_list):
        start = time.time()
        mt = train_model_tfidf(mt, tfidf_train, labels)
        print(f"{name} + TF-IDF trained in {time.time() - start:.2f}s")
        trained_tfidf.append(mt)

        start = time.time()
        mg = train_model_glove(mg, glove_reps, documents, labels)
        print(f"{name} + GloVe trained in {time.time() - start:.2f}s")
        trained_glove.append(mg)

    print("\nTesting models...")
    test_documents, test_labels = load_as_list("test.csv")

    with open("classification_report.csv", "w", newline='\n') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Model", "Precision", "Recall", "F1", "Accuracy"])
        for name, mt, mg in zip(model_names, trained_tfidf, trained_glove):
            p, r, f, a = test_model_tfidf(mt, vectorizer, test_documents, test_labels)
            writer.writerow([name + " + TF-IDF", p, r, f, a])
            p, r, f, a = test_model_glove(mg, glove_reps, test_documents, test_labels)
            writer.writerow([name + " + GloVe", p, r, f, a])

    print("Results saved to classification_report.csv")

    user_input = input("\nWelcome to the sentiment chatbot! What do you want to talk about today?\n")
    glove_test = string2vec(glove_reps, user_input)
    label = mlp_glove.predict(glove_test.reshape(1, -1))

    if label == 0:
        print("Hmm, it seems like you're feeling a bit down.")
    elif label == 1:
        print("It sounds like you're in a positive mood!")
