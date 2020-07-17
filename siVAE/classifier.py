import os
import time
import logging

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from .data.data_handler import kfold_split


def run_classifier(X_train, X_test, y_train, y_test, classifier = "KNeighborsClassifier", max_dim = 1e6, args = {}):
    if classifier == "KNeighborsClassifier":
        clf = KNeighborsClassifier(**args)
    elif classifier == "LogisticRegression":
        clf = LogisticRegression(**args)
    elif classifier == "MLPClassifier":
        clf = MLPClassifier(**args)
    else:
        raise Exception("Input valid classifier")
    _ = clf.fit(X_train,y_train)
    score = clf.score(X_test,y_test)
    return score, clf


# classifier.run_classifiers_over_params(X_classifier,y,k_split,clf_types,classifier_args_dict,classifier_args_ps_dict)

def run_classifiers_over_params(X,y, k_split, clf_types,
                                classifier_args_dict, classifier_args_ps_dict,
                                logdir = "", random_seed = 0, save_npy = False):
    """
    X: matrix
    y: label
    data_splits:
    """

    data_splits = kfold_split(X, y, k_split, random_seed)

    scores = []

    for i_clf, clf_type in enumerate(clf_types):
        print("")
        print("================ Running {} ======================".format(clf_type))

        args = dict(classifier_args_dict[clf_type])

        parameter_dict = classifier_args_ps_dict[clf_type]

        for param_name, param_list in parameter_dict.items():

            npy_scores_param = os.path.join(logdir,"scores_{}.npy".format(clf_type))

            if os.path.isfile(npy_scores_param) and False:
                scores_param = np.load(npy_scores_param)
            else:
                ## Choose the best param
                scores_param = []

                for i_param, param in enumerate(param_list):
                    print("{} = {}".format(param_name, param))
                    args[param_name] = param
                    scores_kfold = np.zeros(len(data_splits))
                    clf_best = None

                    for i_kfold, data_split in enumerate(data_splits):
                        # print("kfold = {}".format(i_kfold))
                        X_train, X_test, y_train, y_test = data_split
                        score, clf = run_classifier(X_train = X_train,
                                                    X_test = X_test,
                                                    y_train = y_train,
                                                    y_test = y_test,
                                                    classifier = clf_type,
                                                    args = args)
                        scores_kfold[i_kfold] = score

                    score_kfold = scores_kfold.mean()
                    scores_param.append(scores_kfold)

                scores_param = np.array(scores_param)
                score_best_param = scores_param[np.argmax(scores_param.mean(-1))]

                if save_npy:
                    np.save(npy_scores_param, np.array(score_best_param))

            print(score_best_param)

        scores.append(score_best_param)

    return scores
