import sys
import argparse
import utilities as ut

import numpy as np
import pylab as pl

from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression    
from sklearn import model_selection
from sklearn import metrics

import matplotlib
matplotlib.style.use('ggplot')


#### FEATURE SELECTION FORWARD PASS
def evaluate_forward_pass(clf, X, y, fs="pearson"): 
    # GET FEATURES RANK
    if fs in ["pearson", "fisher"]:
        ft_ranks, scores = ut.rank_features(np.array(X), y, corr='pearson')
        scores, selected_features = ut.compute_feature_curve(clf, X, y, ft_ranks=ft_ranks, step_size=3)

    elif fs == "greedy":
        # Greedy selection
        scores, selected_features = ut.greedy_selection(clf, X, y)

    return (scores, selected_features)

#### ENSEMBLE FORWARD PASS
def ensemble_forward_pass(clfs, X, y, n_clfs=None):
    if n_clfs == None:
        n_clfs= len(clfs)

    clf_list = ensemble_clfs(clfs)
    auc_scores = np.zeros(n_clfs)

    for i in range(n_clfs):
        skf = model_selection.StratifiedKFold(n_splits=4)

        # CROSS VALIDATE
        scores = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf_list.fit(X_train, y_train, i)
            y_pred = clf_list.predict(X_test)

            scores += [metrics.roc_auc_score(y_test, y_pred)]

        auc_scores[i] = np.mean(scores)
        print "Score: %.3f, n_clfs: %d" % (auc_scores[i], i+1)

    return auc_scores, np.arange(n_clfs) + 1

class ensemble_clfs:
    def __init__(self, clf_list):
        self.clf_list = clf_list
        self.n_clfs = len(clf_list)
        self.trained_clfs = [None] * self.n_clfs
        self.trained_ids = [] 
       

    def fit(self, X, y, clf_id):
        clf = self.clf_list[clf_id]
        clf.fit(X, y)
        self.trained_clfs[clf_id] = clf
        self.trained_ids += [clf_id]

    def predict(self, X):
        n_trained = len(self.trained_clfs)
        pred_list = np.zeros((X.shape[0], n_trained)) 

        for i in self.trained_ids:
            clf = self.trained_clfs[i]

            y_pred = clf.predict_proba(X)[:, 1]
            pred_list[:, i] = y_pred

        return np.mean(pred_list, axis=1)

if __name__ == "__main__":
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    
    # DATASETS
    # 'KC3', 'PC2', 'PC4', 'ant', 'camel', 'MC1'
    parser.add_argument('-fs', '--fs_functions', nargs="+", default=["pearson"])
    parser.add_argument('-m', '--method', default="forward_selection")
    parser.add_argument('-d', '--dataset_name', default="ant")
    parser.add_argument('-n', '--n_clfs', default=5, type=int)

    args = parser.parse_args()      
    method = args.method
    dataset_name = args.dataset_name
    fs_functions = args.fs_functions
    n_clfs = args.n_clfs

    print "\nDATASET: %s\nMETHOD: %s\n" % (dataset_name, method)
    np.random.seed(1)

    # 1. GET DATASET
    X, y, ft_names = ut.read_dataset("datasets/", dataset_name=dataset_name)
    pl.title(dataset_name)
    pl.ylabel("AUC")


    # 2. SELECT EVALUATION METHOD
    if method == "forward_selection":
        """
        Forward selection using weighted svm w.r.t greedy, pearson and fisher
        
        Description in section 5.1 - Results in Fig. 9
        """
        w_svm = SVC(class_weight='balanced', probability=True)

        for fs in fs_functions:
            print "FEATURE SELECTION: %s\n" % fs
            (scores, selected_features) = evaluate_forward_pass(w_svm, X, y, fs)
            pl.plot(selected_features, scores, label=fs)
            
        pl.xlabel("Number of retained features")
            
    elif method == "ensemble_svm":
        """
        Description in section 5.3 - Results in Fig. 10
        """
        clfs = []
        for c in [1, 10, 100, 500, 1000]:
            for w in [{1: 5}, {1: 10}, {1: 15}, {1: 20}, {1: 25}]:
                clfs += [SVC(probability=True, C=c, class_weight=w)]

        (scores, x_values) = ensemble_forward_pass(clfs, X, y, n_clfs=n_clfs)
        pl.plot(x_values, scores, label="weighted-svm ensemble")

    elif method == "ensemble_heter":
        """
        Description in section 5.4 - Results in Fig. 11
        """
        clfs = [SVC(probability=True), MultinomialNB(alpha=0.001),
                BernoulliNB(alpha=0.001), RandomForestClassifier(n_estimators=20),
                GradientBoostingClassifier(n_estimators=300),
                SGDClassifier(alpha=.0001, loss='log', n_iter=50,
                penalty="elasticnet"), LogisticRegression(penalty='l2')]

        (scores, x_values) = ensemble_forward_pass(clfs, X, y, n_clfs=n_clfs)
        pl.plot(x_values, scores, label="heterogenuous ensemble")

    else:
        print "%s does not exist..." % method
        raise

    pl.legend(loc="best")
    pl.show()
