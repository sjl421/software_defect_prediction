import numpy as np
import pylab as pl

from sklearn.model_selection import cross_val_score
from scipy.stats.stats import pearsonr
from sklearn.metrics import mutual_info_score
import pandas as pd
from scipy.sparse import issparse
import ml_metrics
from sklearn import metrics
from scipy.io import arff
from sklearn.metrics import recall_score

##### READING DATASET
def read_dataset(directory, dataset_name):

    if dataset_name in ["ant", "camel"]:
        X = pd.read_csv(directory + dataset_name + '.csv')
        y = X['bug']
        del X['bug']

    elif dataset_name in ["KC3", "PC2", "PC4", "MC1"]:
        data, meta = arff.loadarff(directory + dataset_name + '.arff')
        X =  pd.DataFrame(data)

        y = X['Defective']
        y = mapit(y)
        del X['Defective']

    else:
        print "dataset %s does not exist" % dataset_name


    return np.array(X), np.array(y), []

#### FEATURE SELECTION
def compute_feature_curve(clf, X, y, ft_ranks, step_size=1):
    """plots learning curve """
    selected_features = []
    scores = []

    n_features =  X.shape[1]

    for ft_list in range(step_size, n_features + 1, step_size):
        score = np.mean(cross_val_score(clf, X[:, ft_ranks[:ft_list]], y, 
                                        cv=10, scoring='roc_auc'))

        selected_features += [ft_list]
        scores += [score]

        print 'Score: %.3f with %s features...' % (score, ft_list)

    print 'Best score achieved : %.3f \n' % np.amax(scores)

    return (scores, selected_features)

def greedy_selection(clf, X, y):
    """Applies greedy forward selection"""
    n_features = X.shape[1]

    global_max = 0.0
    selected_features = []
    scores = []

    for i in range(n_features):
        maximum = 0.0
        for j in range(n_features):
            if j in selected_features:
                continue

            score = np.mean(cross_val_score(
                            clf, X[:, selected_features + [j]], y, cv=4,
                            scoring='roc_auc'))

            if score > maximum:
                maximum = score
                best_feature = j

        scores += [score]
        selected_features += [best_feature]

        print 'Score: %.3f with features: %s ...' % (score, selected_features)

        if maximum > global_max:
            global_max = maximum
            #best_features = [f for f in selected_features]

    return scores, selected_features


def rank_features(X, y, corr='fisher'):
    """returns ranked indices using a correlation
         function
    """
    correlation_functions = {
        'fisher': fisher_crit,
        'mutual_info': mutual_info_score,
        'info_gain': information_gain
    }

    results = []

    n_features = X.shape[1]

    if corr in ['pearson']:
        for feature in range(n_features):
            results.append((feature, abs(pearsonr(X[:, feature], y)[0])))

    elif corr in ["fisher"]:
        for feature in range(n_features):
            results.append(
                (feature, correlation_functions[corr](X[:, feature], y)))

    results = sorted(results, key=lambda a: -a[1])

    rank_list = [f[0] for f in results]
    scores = [f[1] for f in results]

    return rank_list, scores

#### MISC
def mapit(vector):

    s = np.unique(vector)

    mapping = pd.Series([x[0] for x in enumerate(s)], index = s)
    vector=vector.map(mapping)
    return vector

def fisher_crit(v1, v2):
    """computes the fisher's criterion"""
    if issparse(v1):
        v1 = v1.todense()
    return abs(np.mean(v1) - np.mean(v2)) / (np.var(v1) + np.var(v2))


def information_gain(v1, v2):
    """computes the information gain"""
    if issparse(v1):
        v1 = v1.todense()
    return abs(np.mean(v1) - np.mean(v2)) / (np.var(v1) + np.var(v2))

#### SCORING METHODS
def g_mean_metric(y_true, y_pred):
    y_pred = np.array([1 if x >= 0.5 else 0 for x in y_pred])

    recall = recall_score(y_true, y_pred)

    i = np.where(y_pred == 0)[0]
    i2 = np.where(y_true == 0)[0]
    tn = float(np.intersect1d(i, i2).size)

    i = np.where(y_pred == 1)[0]
    i2 = np.where(y_true == 0)[0]
    fp = float(np.intersect1d(i, i2).size)

    specifity = (tn / (tn + fp))

    mult = recall * specifity

    return np.sqrt(mult)

def forward_auc(labels, predictions):
    target_one = [1 if x == 1 else 0 for x in labels]
    score = ml_metrics.auc(target_one, predictions)
    return score


def reverse_auc(labels, predictions):
    target_neg_one = [1 if x == -1 else 0 for x in labels]
    neg_predictions = [-x for x in predictions]
    score = ml_metrics.auc(target_neg_one, neg_predictions)
    return score


def bidirectional_auc(labels, predictions):
    score_forward = forward_auc(labels, predictions)
    score_reverse = reverse_auc(labels, predictions)
    score = (score_forward + score_reverse) / 2.0
    return score