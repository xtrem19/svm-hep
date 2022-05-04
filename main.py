# main module
import sys

from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import auc,roc_auc_score

# frame work includes
from data_preparation import data_preparation
import data_utils as du
from mva_utils import variable_importance
import plotting

import pandas as pd
import numpy as np


if len(sys.argv) != 2:
    sys.exit("Provide data sample name. Try again!")

sample_input = sys.argv[1]
data = data_preparation(drop_class=False)

X_train, Y_train, X_test, Y_test = \
    data.dataset(sample_name=sample_input,
                 sampling=False,split_sample=0.3)

species = "class"

model = SVC(C=50, gamma=0.01, kernel='rbf', shrinking = True, probability = False, tol = 0.001)
model.fit(X_train.drop(species, axis=1), Y_train)
Y_pred_dec = model.decision_function(X_test.drop(species, axis=1))
auc = roc_auc_score(Y_test, Y_pred_dec)
print("Testing accuracy", auc)

importance = variable_importance(model, sample_input, X_train.drop(species, axis=1), Y_train, X_test.drop(species, axis=1), Y_test, global_auc=auc, roc_area="deci")
importance.one_variable_out()
importance.recursive_one_variable_out()
importance.permutation_feature()
importance.shapley_values()

over = plotting.Overtraining()
column="output"
data, train_mask, test_mask, signal_mask, bckgrd_mask = du.get_belle2_plot_elements(model, X_train, X_test, species=species)
over.add(data, column, train_mask, test_mask, signal_mask, bckgrd_mask, weight_column=None)
over.save("./figures/overtraining_"+sample_input+".png")
