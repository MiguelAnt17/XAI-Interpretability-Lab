# Utils
import pandas as pd
import numpy as np
import re
import string
from tqdm import tqdm
import matplotlib.pyplot as plt

# Pre-processing
import emoji 
import nltk
from nltk.corpus import stopwords
import spacy 

# Models
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score as sk_accuracy, \
                            precision_score as sk_precision, \
                            recall_score as sk_recall, \
                            roc_auc_score as sk_roc_auc, \
                            classification_report

# Interpretability
from lime.lime_text import LimeTextExplainer


# Average number of words per text
def avgSizeWords(text):
     list_string = text.split()
     if not list_string:
         return 0
     chars = np.array([len(s) for s in list_string])
     return chars.mean()


# Truncate text
def trucateText(text):
    words = text.split()
    if len(words) <= 100:
        return text
    else:
        words = words[0:100]
        text = ' '.join(words)
        return text


#emojis and punctuation
emojis_list = list(emoji.EMOJI_DATA.keys())
emojis_list += ['\n']
punct = list(string.punctuation) + ['\n']
emojis_punct = emojis_list + punct


# function that separates/remove punctuation and emojis
def processEmojisPunctuation(text, remove_punct = False, remove_emoji = False):
    chars = set(text)
    for c in chars:

        if remove_punct:
            if c in punct:
                text = text.replace(c, ' ')
        else:
            if c in punct:
                text = text.replace(c, ' ' + c + ' ')

        if remove_emoji:
            if c in emojis_list:
                text = text.replace(c, ' ')
        else:
            if c in emojis_list:
                text = text.replace(c, ' ' + c + ' ')

    text = re.sub(' +', ' ', text)
    return text

# stop words removal
stop_words = list(stopwords.words('portuguese'))
    new_stopwords = ['aí','pra','vão','vou','onde','lá','aqui',
                    'tá','pode','pois','so','deu','agora','todo',
                    'nao','ja','vc', 'bom', 'ai','ta', 'voce', 'alguem', 'ne', 'pq',
                    'cara','to','mim','la','vcs','tbm', 'tudo','mst', 'ip', 've', 
                    'td', 'msg', 'abs', 'ft', 
                    'rs', 'sqn', 'cmg', 
                    '03', '27', 
                    'http', 'https', 'www',
                    'tocantim']

stop_words = stop_words + new_stopwords
final_stop_words = []
for sw in stop_words:
    sw = ' '+ sw + ' '
    final_stop_words.append(sw)

def removeStopwords(text):
    for sw in final_stop_words:
        text = text.replace(sw,' ')
    text = re.sub(' +',' ',text)
    return text

# lemmatization
nlp = spacy.load('pt_core_news_sm')
def lemmatization(text):
    doc = nlp(text)
    lemmatized_tokens = []
    for token in doc:
        if token.is_punct or token.is_space:
             lemmatized_tokens.append(token.text)
        else:
             lemmatized_tokens.append(token.lemma_)
    return " ".join(lemmatized_tokens)



# the URL are convert to only their domain
'''def domainUrl(text):
    if 'http' in text:
        re_url = '[^\s]*https*://[^\s]*'
        matches = re.findall(re_url, text, flags=re.IGNORECASE)
        for m in matches:
            domain = m.split('//')
            domain = domain[1].split('/')[0]
            text = re.sub(re_url, domain, text, 1)
        return text
    else:
        return text'''


# the URL are removed
def domainUrl(text):
    if 'http' in text:
        re_url = '[^\s]*https*://[^\s]*'
        matches = re.findall(re_url, text, flags=re.IGNORECASE)
        for m in matches:
            text = text.replace(m, ' ')
        text = re.sub(' +', ' ', text).strip()
        return text
    else:
        return text

# remove kkk+ portugueses "stopword"
def processLoL(text):
    re_kkk = 'kkk*'
    t = re.sub(re_kkk, "kkk", text, flags=re.IGNORECASE)
    return t

# extract the first sentence of a text input
def firstSentence(text):
    list_s = re.split('; |\. |\! |\? |\n',text)
    for s in list_s:
        if s is not None:
            return s
    
# miswritten words
correction_map = {
    'olher': 'olhar', 
    'erraddad': 'errado'    
}

def manual_correction(text, mapping):
    for wrong, right in mapping.items():
        text = re.sub(r'\b' + re.escape(wrong) + r'\b', right, text)
    return text

# applies all pre processing functions
def preprocess(text,semi=False, rpunct = False, remoji = False, sentence = False):
    if sentence:
        text = firstSentence(text)
    text = text.lower().strip()
    text = manual_correction(text, correction_map)
    text = domainUrl(text)
    text = processLoL(text)
    text = processEmojisPunctuation(text,remove_punct = rpunct, remove_emoji=remoji)
    if semi:
        return text
    text = removeStopwords(text)
    text = lemmatization(text)
    return text


# Vectorization
def defineVectorizing(experiment):
    max_feat = None
    # maximum number of features
    if 'max_features' in experiment:
        max_feat = 5000
    exp_parts = experiment.split('-')
    vec = exp_parts[0]
    ngram = exp_parts[1]
    # ngram
    if ngram == 'unigram':
        ng = (1,1)
    elif ngram == 'unigram_bigram':
        ng = (1,2)
    elif ngram == 'unigram_bigram_trigram':
        ng = (1,3)

    # n-grams that appear less than five times are not counted
    MIN_FREQUENCY = 5

    # vectorizer
    if vec == 'bow':
        vectorizer = CountVectorizer(max_features = max_feat, binary=True, ngram_range = ng, lowercase = False, token_pattern = r'\b\w\w+\b', min_df=MIN_FREQUENCY)
    elif vec == 'tfidf':
        vectorizer = TfidfVectorizer(max_features = max_feat, ngram_range = ng, lowercase = False, token_pattern = r'\b\w\w+\b', min_df=MIN_FREQUENCY)

    return vectorizer

def vectorizing(vectorizer,texts_train,texts_test):
    vectorizer.fit(texts_train) # learns the vocabulary
    X_train = vectorizer.transform(texts_train) # converts new text into vectors using the learning vocabulary
    X_test = vectorizer.transform(texts_test)
    print('Train:',X_train.shape)
    print('Test:',X_test.shape)
    return X_train, X_test


# extract evaluation metrics
def getTestMetrics(y_true, y_pred, y_prob=None, full_metrics=False, class_names=None):
    """
    Calcula métricas de teste e retorna o relatório de classificação como string.
    """
    # [Cálculo das métricas existentes...]
    acc = sk_accuracy(y_true, y_pred)
    precision = sk_precision(y_true, y_pred, average='weighted')
    recall = sk_recall(y_true, y_pred, average='weighted')
    epsilon = 1e-7
    f1 = 2 * (precision * recall) / (precision + recall + epsilon) if (precision + recall) > 0 else 0

    try:
        roc_auc = sk_roc_auc(y_true, y_prob, multi_class='ovr')
    except Exception:
        roc_auc = np.nan

    precision_neg = recall_neg = f1_neg = np.nan
    
    # 🎯 Gerar o Classification Report como string
    report_str = classification_report(y_true, y_pred, target_names=class_names, output_dict=False)
    
    if full_metrics:
        print(f"## 📊 Métricas de Desempenho (Weighted) ##")
        print(f"Accuracy: {acc:.3f}")
        print(f"Precision (W): {precision:.3f}")
        print(f"Recall (W): {recall:.3f}")
        print(f"F1 (W): {f1:.3f}")
        print(f"AUC: {roc_auc:.3f}")
        print("\n---")
        print("## 📋 Classification Report ##")
        print(report_str)

    # 🎯 Agora retorna o relatório como o último elemento
    return acc, precision, precision_neg, recall, recall_neg, f1, f1_neg, roc_auc, report_str


# Models Training & Evaluation
def lr_eval(X_train, y_train, X_test, y_test):
    print('=== Logistic Regression ===')
    logreg = LogisticRegression().fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    y_prob = logreg.predict_proba(X_test)[:, 1]
    metrics = getTestMetrics(y_test, y_pred, y_prob, full_metrics=True)
    return logreg, metrics


def nb_eval(X_train, y_train, X_test, y_test, experiment):
    if 'bow' in experiment[0]:
        print('=== Bernoulli Naive-Bayes ===')
        nb = BernoulliNB().fit(X_train, y_train)
    elif 'tfidf' in experiment[0]:
        print('=== Complement Naive-Bayes ===')
        nb = ComplementNB().fit(X_train, y_train)
    else:
        nb = BernoulliNB().fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    y_prob = nb.predict_proba(X_test)[:, 1]
    metrics = getTestMetrics(y_test, y_pred, y_prob, full_metrics=True)
    return nb, metrics


def lsvm_eval(X_train, y_train, X_test, y_test):
    print('=== Linear Support Vector Machine ===')
    svm = LinearSVC(dual=False).fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    metrics = getTestMetrics(y_test, y_pred, full_metrics=True)
    return svm, metrics


def sgd_eval(X_train, y_train, X_test, y_test):
    print('=== Linear SVM with SGD training ===')
    sgd = SGDClassifier().fit(X_train, y_train)
    y_pred = sgd.predict(X_test)
    metrics = getTestMetrics(y_test, y_pred, full_metrics=True)
    return sgd, metrics


def svm_eval(X_train, y_train, X_test, y_test):
    print('=== SVM with RBF kernel ===')
    svc = SVC(probability=True).fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    y_prob = svc.predict_proba(X_test)[:, 1]
    metrics = getTestMetrics(y_test, y_pred, y_prob, full_metrics=True)
    return svc, metrics


def knn_eval(X_train, y_train, X_test, y_test):
    print('=== KNN ===')
    knn = KNeighborsClassifier(weights='distance', n_jobs=-1).fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_prob = knn.predict_proba(X_test)[:, 1]
    metrics = getTestMetrics(y_test, y_pred, y_prob, full_metrics=True)
    return knn, metrics


def rf_eval(X_train, y_train, X_test, y_test):
    print('=== Random Forest ===')
    rf = RandomForestClassifier(n_jobs=-1).fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    metrics = getTestMetrics(y_test, y_pred, y_prob, full_metrics=True)
    return rf, metrics


def gb_eval(X_train, y_train, X_test, y_test):
    print('=== Gradient Boosting ===')
    gb = GradientBoostingClassifier(n_estimators=200).fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    y_prob = gb.predict_proba(X_test)[:, 1]
    metrics = getTestMetrics(y_test, y_pred, y_prob, full_metrics=True)
    return gb, metrics


def mlp_eval(X_train, y_train, X_test, y_test):
    print('=== Multilayer Perceptron ===')
    mlp = MLPClassifier(
        verbose=True, early_stopping=True,
        batch_size=64, n_iter_no_change=5, tol=1e-3
    ).fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    y_prob = mlp.predict_proba(X_test)[:, 1]
    metrics = getTestMetrics(y_test, y_pred, y_prob, full_metrics=True)
    return mlp, metrics


def model_eval(model, X_train, y_train, X_test, y_test, experiment=None):
    if model == 'lr':
        return lr_eval(X_train, y_train, X_test, y_test)
    elif model == 'nb':
        return nb_eval(X_train, y_train, X_test, y_test, experiment)
    elif model == 'lsvm':
        return lsvm_eval(X_train, y_train, X_test, y_test)
    elif model == 'sgd':
        return sgd_eval(X_train, y_train, X_test, y_test)
    elif model == 'svm':
        return svm_eval(X_train, y_train, X_test, y_test)
    elif model == 'knn':
        return knn_eval(X_train, y_train, X_test, y_test)
    elif model == 'rf':
        return rf_eval(X_train, y_train, X_test, y_test)
    elif model == 'gb':
        return gb_eval(X_train, y_train, X_test, y_test)
    elif model == 'mlp':
        return mlp_eval(X_train, y_train, X_test, y_test)
    else:
        raise ValueError(f"Model '{model}' unknown.")
    


# Jaccard
def calculate_jaccard(set_a, set_b):
    if not isinstance(set_a, set): set_a = set(set_a)
    if not isinstance(set_b, set): set_b = set(set_b)
        
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    
    if union == 0:
        return 0.0
    return intersection / union



# Plot for Comparison of Models
def plot_models_metrics(models_results, save_path='model_comparison.png'):
    """Cria gráfico de comparação de métricas dos modelos"""
    
    # Extrair dados das métricas
    model_names = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for name, (model, metrics) in models_results.items():
        model_names.append(name)
        accuracies.append(metrics[0])   # accuracy
        precisions.append(metrics[1])   # precision
        recalls.append(metrics[3])      # recall
        f1_scores.append(metrics[5])    # f1
    
    # Criar DataFrame
    df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1-Score': f1_scores
    })
    
    # Configurações do gráfico
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Definir posições das barras
    x = np.arange(len(df['Model']))
    width = 0.2
    
    # Paleta de verdes
    colors = ['#90EE90', '#66CDAA', '#3CB371', '#2E8B57']
    
    # Criar as barras
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = width * (i - 1.5)
        ax.bar(x + offset, df[metric], width, label=metric, color=color, 
               edgecolor='black', linewidth=0.7, alpha=0.9)
    
    # Configurações
    ax.set_xlabel('Models', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], fontsize=10)
    ax.set_ylim(0.6, 0.75)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9, ncol=4)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{save_path}'")
    
    plt.show()
    
    print("\nMetrics table:")
    print(df.to_string(index=False))
    
    return fig, ax, df