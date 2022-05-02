import numpy as np
import pandas as pd


def roc_curve_adaboost(Y_thresholds, Y_test):
    # function to create the TPR and FPR, for ROC curve

    # check data format
    if type(Y_test) != type(np.array([])):
        Y_test = Y_test.values

    TPR_list, FPR_list = [], []
    for i in range(Y_thresholds.shape[0]):
        tp,fn,tn,fp=0,0,0,0
        for j in range(Y_thresholds.shape[1]):             
            if(Y_test[j] == 1  and Y_thresholds[i][j] ==  1):  tp+=1
            if(Y_test[j] == 1  and Y_thresholds[i][j] == -1):  fn+=1
            if(Y_test[j] == -1 and Y_thresholds[i][j] == -1):  tn+=1
            if(Y_test[j] == -1 and Y_thresholds[i][j] ==  1):  fp+=1

        TPR_list.append( tp/(tp+fn) )
        FPR_list.append( fp/(tn+fp) )

    # sort the first list and map ordered indexes to the second list
    FPR_list, TPR_list = zip(*sorted(zip(FPR_list, TPR_list)))
    TPR = np.array(TPR_list)
    FPR = np.array(FPR_list)

    return TPR,FPR

def get_belle2_plot_elements(model, X_train, X_test, species="class"):
    # yields the elements needed to call the plotting functions borrowed from basf2
    
    X_train_sig = X_train[X_train[species]==+1]
    X_train_bkg = X_train[X_train[species]==-1]
    X_test_sig  = X_test[X_test[species]==+1]
    X_test_bkg  = X_test[X_test[species]==-1]
    
    d_sig_train = model.decision_function(X_train_sig.drop(species, axis=1))
    d_bkg_train = model.decision_function(X_train_bkg.drop(species, axis=1))
    d_sig_test  = model.decision_function(X_test_sig.drop(species, axis=1))
    d_bkg_test  = model.decision_function(X_test_bkg.drop(species, axis=1))
    
    data_tot = np.array([])
    data_tot = np.append(data_tot, d_sig_train)
    data_tot = np.append(data_tot, d_bkg_train)
    data_tot = np.append(data_tot, d_sig_test)
    data_tot = np.append(data_tot, d_bkg_test)
    col_out = pd.DataFrame(data=data_tot,    columns=["output"])
    data = pd.concat([col_out["output"]], axis=1, keys=["output"])
    
    train_mask  = np.full((len(data_tot)), False)
    test_mask   = np.full((len(data_tot)), False)
    signal_mask = np.full((len(data_tot)), False)
    bckgrd_mask = np.full((len(data_tot)), False)
    
    end_train_sig_index  = len(d_sig_train)
    start_test_sig_index = len(d_sig_train) + len(d_bkg_train)
    end_test_sig_index   = len(d_sig_train) + len(d_bkg_train) + len(d_sig_test)
    
    for i in range(len(train_mask)):
        if i<start_test_sig_index:
            train_mask[i] = True
        else:
            test_mask[i] = True
        if i<end_train_sig_index or (i>start_test_sig_index-1 and i<end_test_sig_index):
            signal_mask[i] = True
        if (i>=end_train_sig_index and i<start_test_sig_index) or i>=end_test_sig_index:
            bckgrd_mask[i] = True

    return data, train_mask, test_mask, signal_mask, bckgrd_mask
