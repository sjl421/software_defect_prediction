from sklearn.metrics import fbeta_score, make_scorer
import numpy as np
import pylab as pl
from sklearn import cross_validation
from scipy.stats.stats import pearsonr
from sklearn.metrics import mutual_info_score
import pandas as pd
from scipy.sparse import issparse
import ml_metrics
from sklearn import metrics

def read_arff(f):
    from scipy.io import arff
    data, meta = arff.loadarff(f)
    return pd.DataFrame(data)

def g_mean_metric(y_true, y_pred):

    y_pred = np.array([1 if x >= 0.5 else 0 for x in y_pred])

    # estimator.fit(X,y_true)
    #y_pred = estimator.predict(X)
    from sklearn.metrics import recall_score
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

def read(st):
    return read_nasa('D:/Datasets/defectDatasets/' + st + '.arff')

def read_nasa(data):
    X = read_arff(data)
    y = X['Defective']
    y = mapit(y)
    del X['Defective']

    return X, y

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

def auc_score_multiclass(clf, X, y):
    X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
        X, y, test_size=0.3, random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_cv).flatten()
    score = bidirectional_auc(y_cv, y_pred)
    return score

def greedy_selection(clf, X, y, score_func='accuracy', datam=False, lol=False,
                     plot=False, getAUC=False, scoring='auc'):
    """applies greedy forward selection"""
    n_classes = len(np.unique(y))
    n_features = X.shape[1]
    # print 'Total # of features', n_features
    best_features = []
    global_max = 0.0
    selected_features = []
    retained_scores = []
    from sklearn.metrics import fbeta_score, make_scorer
    g_mean_metric_scorer = make_scorer(g_mean_metric)
    feature_score = []
    for i in range(n_features):
        maximum = 0.0
        for j in range(n_features):
            if j in selected_features:
                continue
            if n_classes > 2:
                score = auc_score_multiclass(
                    clf, X[:, selected_features + [j]], y)
            else:
                if isinstance(clf, list):
                    # print 'hahahhaFeatures_Select_Defect'
                    score = vote(
                        clf, X[:, selected_features + [j]], y,
                        geometric_mean=True)
                elif scoring == 'auc':
                    score = np.mean(cross_validation.cross_val_score(
                        clf, X[:, selected_features + [j]], y, cv=4,
                        scoring='roc_auc'))
                elif scoring == 'gmeans':
                    score = np.mean(cross_validation.cross_val_score(
                        clf, X[:, selected_features + [j]], y, cv=4,
                        scoring=g_mean_metric_scorer))
                # print score

                # werewr
                # score=testClassification(clf,X[:,selected_features+[j]],
                # y,getValues=True, scoring = scoring)
            if score > maximum:
                maximum = score
                best_feature = j
        retained_scores.append(score)
        # print 'score: ', score, '# features: ',len(selected_features)+1
        selected_features.append(best_feature)
        # added for printing features,score
        #asd = ', '.join(datam.columns[selected_features].tolist())

        # feature_score.append([asd,score])
        # print asd
        if maximum > global_max:
            global_max = maximum
            best_features = [f for f in selected_features]
    c = best_features
    c.append(-1)
    # added for printing features,scoreb[b.columns[0]]
    #a = DataFrame(datam.ix[:,datam.columns[c]])
    #a['scores'] = Series(score)
    # a.to_csv('D:\\Datasets\\random_forest_results_'+lol+'.csv')
    #np.save('D:\\Datasets\\results.npy', feature_score)

    # print 'Best score: ', global_max, 'with',len(best_features), 'features',
    # 'features:', best_features
    print 'Best score: ', global_max
    # return global_max
    # plot retained scores
    if plot:
        pl.plot(np.arange(n_features),
                   retained_scores, marker='o', linestyle='-')
        pl.ylabel(score_func)
        pl.xlabel('Number of features')
        pl.show()

    return best_features


def Ensemble_Defect(type_='heter', indices=False, scoring='auc'):
    print '===================================='
    from sklearn.svm import SVC, LinearSVC
    from sklearn import svm, grid_search, datasets
    from sklearn.ensemble import RandomForestClassifier
    np.random.seed(1)
    i = 0
    for data, cat in [('KC3', 0), ('PC2', 0), ('PC4', 0), ('ant', 1),
                      ('camel', 1), ('MC1', 0)]:
        clfs = clfs_defect(type_=type_)
        if cat == 0:
            X, y = read(data)
        else:
            X, y = read_csv_defect(data)
        feature_names_ = np.array(X.columns)

        X = np.array(X)
        print data
        if scoring == 'auc':
            scorer = 'roc_auc'
        elif scoring == 'gmeans':
            scorer = make_scorer(g_mean_metric)

        #clf = SVC(probability=True)
        # print 'RBF SVM :', np.mean(cross_validation.cross_val_score(clf, X,
        # y, cv=10, scoring=scorer))
        clf = LinearSVC()
        print 'linear SVM :', np.mean(cross_validation.cross_val_score(clf, X,
                                                                       y, cv=10, scoring=scorer))
        clf = SVC(probability=True)
        print 'RBF SVM :', np.mean(cross_validation.cross_val_score(clf, X, y,
                                                                    cv=10, scoring=scorer))
        print 1. / X.shape[1]
        parameters = {'gamma': np.arange(0, 0.1, 0.01)}

        clf = SVC(probability=True)
        clf = grid_search.GridSearchCV(clf, parameters, scoring=scorer)
        clf.fit(X, y)
        print 'Optimized RBF SVM :', clf.best_score_
        continue
        #clf = SVC(kernel = 'linear',probability=True)
        # print 'Linear SVM :', np.mean(cross_validation.cross_val_score(clf,
        # X, y, cv=10, scoring=scorer))

        #clf = SVC(class_weight='auto',probability=True)
        # print 'Weighted SVM :', np.mean(cross_validation.cross_val_score
        #(clf, X, y, cv=10, scoring=scorer))
        #clf = RandomForestClassifier()
        # print 'Random Forest :',  np.mean(cross_validation.cross_val_score
        #(clf, X, y, cv=10, scoring=scorer))
        # print 'Ensemble :',vote(clfs,X,y, scoring = scoring)
        if indices == False:
            classifier_selection(clfs, X, y, name=data)
        # print 'Ensemble with geometric
        # mean:',vote(clfs,X,y,geometric_mean=True)
        if indices == True:
            print 'weighted SVM with selected features :'
            features = greedy_selection(
                SVC(class_weight='auto', probability=True), X, y, getAUC=True,
                scoring=scoring)
            print 'weighted SVM features:', feature_names_[features]

            print 'Random Forest with selected features:'
            features = greedy_selection(
                RandomForestClassifier(), X, y, getAUC=True, scoring=scoring)
            print 'Random Forest features:', feature_names_[features]

            print 'Ensemble with selected features:'
            features = greedy_selection(
                clfs, X, y, getAUC=True, scoring=scoring)
            print 'APE features:', feature_names_[features]
            i += 1
def read_csv_defect(name):
    X = pd.read_csv('E:/Datasets/defectDatasets/' + name + '.csv')
    y = X['bug']
    del X['bug']
    return X, y

def Features_Select_Defect():
    from sklearn.svm import SVC, LinearSVC
    np.random.seed(1)
    step = 3

    bestFeatureSets = []
    scores = []
    # for data,cat in
    # [('KC3',0),('PC2',0),('PC4',0),('ant',1),('camel',1),('MC1',0)]:
    for data, cat in [('ant', 1), ('camel', 1), ('MC1', 0)]:
        print 'running for ', data
        if cat == 0:
            X, y = read(data)
        else:
            X, y = read_csv_defect(data)

        #
        # return 0

        g_mean_metric_scorer = make_scorer(g_mean_metric)
        # X=np.array(X)
        # y=np.array(y)
        clf = SVC(class_weight='auto', probability=True)
        #clf= RandomForestClassifier()
        """
        #score=np.mean(cross_validation.cross_val_score(clf, X, y, cv=3,
         scoring=g_mean_metric_scorer))
        score = greedy_selection(clf,X,y)
        scores.append(['Weighted SVM', data, score])

        clf= RandomForestClassifier()

        #score=np.mean(cross_validation.cross_val_score(clf, X, y, cv=3,
            scoring=g_mean_metric_scorer))
        score = greedy_selection(clf,X,y)
        scores.append(['Random Forest', data, score])
        continue
        DataFrame().to_csv('D:\\Datasets\\'+data+'.csv')
        continue
        """
        #clf= RandomForestClassifier()
        # print clf
        myCsv = X
        # print X.columns
        # added
        real = X.join(y, on=X.columns[0])
        # print real

        X = np.array(X)
        fts = []
        # Pearson's criterion
        fts.append(rank_features(np.array(X), y, corr='pearson'))
        # Fisher's criterion
        fts.append(rank_features(np.array(X), y, corr='fisher'))
        results = []
        ft, bestFeatures = greedy_selection(
            clf, X, y, datam=real, lol=data, plot=False)
        fts.append(ft)
        # print np.array(X[bestFeatures]).shape, np.array(y)
        #[:,np.newaxis].shape
        # DataFrame(np.hstack([np.array(X[:,bestFeatures]),np.array(y)
        # [:,np.newaxis]])).to_csv('D:\\Datasets\\'+data+'.csv')
        results.append(
            feature_curve(clf, X, y, indices=ft, step_size=step, data=data))
        results.append(
            feature_curve(clf, X, y, indices=fts[0], step_size=step, data=data))
        results.append(
            feature_curve(clf, X, y, indices=fts[1], step_size=step, data=data))
        # bestFeatureSets.append(bestFeatures)
        # print 'best features for',data,':',myCsv.columns[bestFeatures]

        plot_results(
            results, descs=['Greedy Selection', 'Pearson Correlation',
                            'Fisher Criterion'], save=data)
    # DataFrame(scores).to_csv('D:\\Datasets\\scores_2.csv')
    return bestFeatureSets

def plot_results(ls, descs=['SLVM-Based SGD', 'CFV-Based SGD'], save=False):
    assert len(ls) == len(descs)

    assert len(ls[0][0]) == len(ls[0][1])
    markers = ['s', 'o', 'D']
    linestyles = ['-', '-.', ':']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i in range(len(ls)):
        pl.plot(ls[0][1], ls[i][0], c=colors[i], marker=markers[
                i], linestyle=linestyles[i], label=descs[i])
    ls = np.array(ls)

    pl.ylim([np.min(ls[:, 0]) - 0.005, np.max(ls[:, 0]) + 0.005])
    pl.ylabel('Accuracy')
    pl.xlabel('Number of retained features')
    pl.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
              ncol=2, mode="expand", borderaxespad=0.)
    if save:
        pl.savefig('D:/Software_Defect/feature_selection/' + save + '.eps',
                   format='eps')
        pl.close()
    else:
        pl.show()

def feature_curve(clf, X, y, indices, data, step_size=1):
    """plots learning curve """
    x_points = []
    scores = []
    n_classes = len(np.unique(y))
    n_samples, n_features = X.shape[0], X.shape[1]
    # X, X_test, y, y_test = cross_validation.train_test_split(X,y,
    #        test_size=0.30,random_state=0)
    # Get scores' correlation with feature size

    for features in range(step_size, n_features + 1, step_size):

        score = np.mean(cross_validation.cross_val_score(
            clf, X[:, indices[:features]], y, cv=10, scoring='roc_auc'))
        """
            y_pred = clf.predict_proba(X_test[:, indices[:features]])[:,1]
            fpr, tpr, thresholds = roc_curve(y_test, y_pred)
            score = auc(fpr, tpr)
            """
        #plot_roc_curve(y_test, y_pred, save = data+'_auc_'+str(score))

        x_points.append(features)
        scores.append(score)
        print 'Score: ', score, 'with', features, 'features'
    print 'Best score achieved :', np.amax(scores)
    return (scores, x_points)

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
    if corr == 'pearson':
        for feature in range(n_features):
            results.append((feature, abs(pearsonr(X[:, feature], y)[0])))
    else:
        for feature in range(n_features):
            results.append(
                (feature, correlation_functions[corr](X[:, feature], y)))

    results = sorted(results, key=lambda a: -a[1])
    return [f[0] for f in results]


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

def vote(estimators, X, y, geometric_mean=False, balanced=False,
         scoring='auc'):

    np.random.seed(1)
    # cross validation

    skf = cross_validation.StratifiedKFold(y, n_folds=4)
    auc_scores = np.zeros(4)
    i = 0
    for train_index, test_index in skf:

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        ls = []

        for clf in estimators:
            clf.fit(X_train, y_train)
            if balanced:
                pred = np.sum(clf.predict_proba(X_test)[:, 1:], axis=1)
            else:
                pred = clf.predict_proba(X_test)[:, 1]
            ls.append(pred)

        s = np.zeros(ls[0].shape)
        for a in ls:
            s += a
        pred_test = s / len(ls)
        if scoring == 'auc':
            auc_scores[i] = metrics.roc_auc_score(y_test, pred_test)
        elif scoring == 'gmeans':
            auc_scores[i] = g_mean_metric(y_test, pred_test)
        i += 1
        # print auc_scores
    return np.mean(auc_scores)