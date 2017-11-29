import sys
import argparse
import utilities as ut

import numpy as np
import pylab as pl


import methods

if __name__ == "__main__":
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    
    # DATASETS
    # 'KC3', 'PC2', 'PC4', 'ant', 'camel', 'MC1'

    # fs_functions: pearson, fisher, greedy_auc, greedy_gmeans

    # fs_functions defines the forward selection functions
    parser.add_argument('-fs', '--fs_functions', nargs="+", required=True,
                        choices=["pearson", "fisher", "greedy"])

    parser.add_argument('-m', '--method', default="forward_selection",
                        choices=["forward_selection",
                                 "ensemble_svm",
                                 "ensemble_heter"])

    parser.add_argument('-d', '--dataset_name', default="ant")

    parser.add_argument('-n', '--n_clfs', default=5, type=int)

    parser.add_argument('-s', '--score_name', required=True, 
                        choices=["auc","gmeans"])

    args = parser.parse_args()      
    method = args.method
    dataset_name = args.dataset_name
    fs_functions = args.fs_functions
    n_clfs = args.n_clfs
    score_name = args.score_name

    print("\nDATASET: %s\nMETHOD: %s\n" % (dataset_name, method))
    np.random.seed(1)


    ##### 1. ------ GET DATASET
    X, y, ft_names = ut.read_dataset("datasets/", dataset_name=dataset_name)
    pl.title(dataset_name)
    pl.ylabel("AUC")

    ##### 2. ------- RUN TRANING METHOD
    methods.run_method(method, X, y, n_clfs=n_clfs, 
                       fs_functions=fs_functions, 
                       score_name=score_name)

    pl.legend(loc="best")
    pl.show()
