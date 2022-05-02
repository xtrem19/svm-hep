# data preparation module
import sys
import pandas as pd
import numpy as np
import uproot # ROOT format data
# sklearn utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
# framework includes
#import data_visualization as dv

class data_preparation:

    def __init__(self, path=".", drop_class=True, GA_selection=False):
        self.m_drop_class = drop_class
        self.workpath = path
        self.genetic = GA_selection

    # fetch data
    def fetch_data(self, sample):
        if sample == "toy":
            df_sig = pd.read_csv(self.workpath+"/data/toy_sig.csv")
            df_bkg = pd.read_csv(self.workpath+"/data/toy_bkg.csv")
            frames = [df_sig, df_bkg]
            data_set = pd.concat(frames)
            # df_data= pd.read_csv(self.workpath+"/data/toy_data.csv")
        elif sample == "belle2_d0":
            file_data = uproot.open(self.workpath+"/data/D0_kpipi0_generic_svm_test.root")
            data_set = file_data["variables"].arrays(library="pd")
        elif sample == "belle2_i":
            file_data = uproot.open(self.workpath+"/data/belle2_kpipi0.root")
            data_set = file_data["combined"].arrays(library="pd")
        elif sample == "belle2_ii":
            file_data = uproot.open(self.workpath+"/data/belle2_kpi.root")
            data_set = file_data["combined"].arrays(library="pd")
        elif sample == "belle2_iii":
            file_train = uproot.open(self.workpath+"/data/train_D02k3pi.root")
            data_train = file_train["d0tree"].arrays(library="pd")
            file_test  = uproot.open(self.workpath+"/data/test_D02k3pi.root")
            data_test  = file_test["d0tree"].arrays(library="pd")
            return (data_train, data_test)
        elif sample == "belle2_iv":
            file_train = uproot.open(self.workpath+"/data/train_D02kpipi0vxVc-cont0p5.root")
            #file_train = uproot.open(self.workpath+"/data/test_belle2_iv.root")
            data_train = file_train["d0tree"].arrays(library="pd")
            file_test  = uproot.open(self.workpath+"/data/test_D02kpipi0vxVc-cont0p5.root")
            data_test  = file_test["d0tree"].arrays(library="pd")
            #return data_train
            return (data_train, data_test)
        elif sample == "belle2_challenge":
            file_train = uproot.open(self.workpath+"/data/train_D02kpipi0vxVc-cont0p5.root")
            data_train = file_train["d0tree"].arrays(library="pd")
            file_test  = uproot.open(self.workpath+"/data/test_D02kpipi0vxVc-cont0p5.root")
            data_test  = file_test["d0tree"].arrays(library="pd")
            return (data_train, data_test)
        elif sample == "titanic":
            data_set = pd.read_csv(self.workpath+"/data/titanic.csv")
        else:
            sys.exit("The sample name provided does not exist. Try again!")
        return data_set


    # call data
    def dataset(self, sample_name, data_set=None, data_train=None, data_test=None,
                sampling=False, split_sample=0, indexes = None):

        train_test = False # to check if data is divided
        # if sampling=True, sampling is done outside,sample is fetched externally

        # fetch data_set if NOT externally provided
        if not sampling:
            data_temp = self.fetch_data(sample_name)
            train_test = type(data_temp) is tuple
            if not train_test:
                data_set = data_temp
            else: # there is separate data samples for training and testing
                data_train,data_test = data_temp
            
        # prepare data
        if sample_name == "toy":
            X,Y = self.toy_data(data_set, sampling, sample_name=sample_name)
        if sample_name == "belle2_d0":
            X,Y = self.belle2_d0(data_set, sampling, sample_name=sample_name)
        elif sample_name == "belle2_i" or sample_name == "belle2_ii":
            X,Y = self.belle2(data_set, sampling, sample_name=sample_name)
        elif sample_name == "belle2_iii":
            train_test = True
            X_train, Y_train, X_test, Y_test = self.belle2_3pi(data_train, data_test, sampling, sample_name=sample_name)
        elif sample_name == "belle2_iv":
            train_test = True
            X_train, Y_train, X_test, Y_test = self.belle2_iv(data_train, data_test, sampling, sample_name=sample_name)
        elif sample_name == "belle2_challenge":
            train_test = True
            X_train, Y_train, X_test, Y_test = self.belle2_challenge(data_train, data_test, sampling, sample_name=sample_name)
        elif sample_name=="titanic":
            X,Y = self.titanic(data_set)

        # print data after preparation
        if not sampling:
            if train_test:
                print(X_train.head())#, Y.head())
                print(Y_train.head())#, Y.head())
            else:
                print(X.head())#, Y.head())
                print(Y.head())#, Y.head())
                
        # return X,Y without any spliting (for bootstrap and kfold-CV)
        if sampling or split_sample==0.0:
            if not train_test:
                return X,Y
            else:
                return X_train, Y_train, X_test, Y_test   
                                  
        # divide sample into train and test sample
        if indexes is None:
            if not train_test:
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_sample, random_state=2) # split_sample
        else:
            if not train_test:
                X_train, X_test, Y_train, Y_test = self.indexes_split(X, Y, split_indexes=indexes, train_test=train_test)
            else:
                X_train, X_test, Y_train, Y_test = self.indexes_split(X_train, Y_train, X_test, Y_test,
                                                                      split_indexes=indexes, train_test=train_test)
                
        return X_train, Y_train, X_test, Y_test

    
    def indexes_split(self, X, Y, x_test=None, y_test=None, split_indexes=None, train_test=False):
        """ Function to split train and test data given train indexes"""
        if not train_test:
            total_indexes = np.array(X.index).tolist()        
            train_indexes = split_indexes.tolist()
            test_indexes  = list(set(total_indexes) - set(train_indexes))
        else:
            train_indexes = split_indexes.tolist()
            test_indexes  = split_indexes.tolist()

        X_train = X.loc[train_indexes]
        Y_train = Y.loc[train_indexes]
        X_test  = X.loc[test_indexes]
        Y_test  = Y.loc[test_indexes]

        return X_train, X_test, Y_train, Y_test
    

    def toy_data(self, data_set=None, sampling=False, sample_name="toy", sig_back='sig'):

        if data_set is None:
            if sig_back=="sig":   data_set = pd.read_csv(self.workpath+"/data/toy_sig.csv")
            elif sig_back=="bkg": data_set = pd.read_csv(self.workpath+"/data/toy_bkg.csv")
            elif sig_back=="data":data_set = pd.read_csv(self.workpath+"/data/toy_data.csv")
            
        if not sampling:
            data_set = resample(data_set, replace = False, n_samples = 5000, random_state = 1)

        data_set = data_set.copy()
        
        Y = data_set["class"]
        # data scaling [0,1]
        # data_set = pd.DataFrame(MinMaxScaler().fit_transform(data_set), # not doing this for gaussian distributions!!
        #                         columns = list(data_set.columns))
        X = data_set #.drop("class", axis=1)
        return X,Y

    def belle2_d0(self, data_set=None, sampling=False, sample_name='belle2_d0', sig_back='both'):

        if data_set is None:
            file_data = uproot.open(self.workpath+"/data/D0_kpipi0_generic_svm_test.root")
            data_set = file_data["variables"].arrays(library="pd")

        data_set = data_set.copy()
        data_set.loc[data_set["isSignal"] != 1, "isSignal"] = -1
        data_set = data_set.drop("__experiment__", axis=1)
        data_set = data_set.drop("__run__", axis=1)
        data_set = data_set.drop("__event__", axis=1)
        data_set = data_set.drop("__candidate__", axis=1)
        data_set = data_set.drop("__ncandidates__", axis=1)
        data_set = data_set.drop("__weight__", axis=1)
        data_set = data_set.drop("M", axis=1)
        data_set = data_set.drop("useCMSFrame__bop__bc", axis=1)
        
        if sig_back=="sig":
            data_set = data_set.loc[data_set["isSignal"] == 1]
        elif sig_back=="bkg":
            data_set = data_set.loc[data_set["isSignal"] == -1]
        elif sig_back=="data":
            data_set = data_set
        else:
            data_set_sig = data_set.loc[data_set["isSignal"] == 1]
            data_set_bkg = data_set.loc[data_set["isSignal"] == -1]

            
        if not sampling:
            if sig_back=="sig" or sig_back=="bkg" or sig_back=="data":
                data_set     = data_set# resample(data_set, replace = False, n_samples = 5000, random_state = None)
            else:
                data_set_sig = resample(data_set_sig, replace = False, n_samples = 5000, random_state = None)
                data_set_bkg = resample(data_set_bkg, replace = False, n_samples = 5000, random_state = None)
                frames = [data_set_sig, data_set_bkg]
                data_set = pd.concat(frames)

        Y = data_set["isSignal"]
        # data scaling [0,1]
        data_set = pd.DataFrame(MinMaxScaler().fit_transform(data_set), # not doing this for gaussian distributions!!
                                columns = list(data_set.columns))

        if self.m_drop_class: 
            X = data_set.drop("isSignal", axis=1)
        else:
            X = data_set

        return X,Y

    
    # belle2 data preparation
    def belle2(self, data_set, sampling, sample_name):
        
        if(sampling or self.genetic): # sampling was already carried, don"t sample again!
            Y = data_set["Class"]
            # Data scaling [0,1]
            # column names list            
            cols = list(data_set.columns)
            data_set = pd.DataFrame(MinMaxScaler().fit_transform(data_set),columns = cols)
            X = data_set.drop("Class", axis=1)
            return X,Y 

        sampled_data = resample(data_set, replace = False, n_samples = 5000, random_state = 0)

        Y = sampled_data["Class"]
        # column names list
        cols = list(sampled_data.columns)
        # Data scaling [0,1]
        #sampled_data = pd.DataFrame(MinMaxScaler().fit_transform(sampled_data),columns = cols)

        if self.m_drop_class:
            X = sampled_data.drop("Class", axis=1)
        else:
            X = sampled_data

        # # plot variables for visualization (now only for HEP)
        # dv.plot_hist_frame(data_set,"full_"+sample_name)
        # dv.plot_hist_frame(sampled_data,"sampled_"+sample_name)            
        return X,Y


    # belle2 data preparation
    def belle2_iv(self, data_train, data_test, sampling, sample_name):
        
        data_train = data_train.copy()
        data_train.loc[data_train["isSignal"] == 0, "isSignal"] = -1
        data_test = data_test.copy()
        data_test.loc[data_test["isSignal"] == 0, "isSignal"] = -1

        if(sampling or self.genetic): # sampling already done or not needed
            Y_train = data_train["isSignal"]
            Y_test  = data_test["isSignal"]
            # Data scaling [0,1]
            cols = list(data_train.columns)        
            data_train = pd.DataFrame(MinMaxScaler().fit_transform(data_train),columns = cols)
            data_train = data_train.drop("vM", axis=1)
            data_train = data_train.drop("vpCMS", axis=1)
            data_train = data_train.drop("__index__", axis=1)

            data_test  = pd.DataFrame(MinMaxScaler().fit_transform(data_test) ,columns = cols)
            data_test  = data_test.drop("vM", axis=1)
            data_test  = data_test.drop("vpCMS", axis=1)
            data_test  = data_test.drop("__index__", axis=1)

            if self.m_drop_class:
                X_test  = data_test.drop("isSignal", axis=1)
                X_train = data_train.drop("isSignal", axis=1)
            else:
                X_test  = data_test
                X_train = data_train
            return X_train, Y_train, X_test, Y_test
        
        sampled_data_train = resample(data_train, replace = False, n_samples = 1000, random_state=None)
        sampled_data_test  = resample(data_test,  replace = False, n_samples = 1000, random_state=None)
        
        Y_train = sampled_data_train["isSignal"]
        Y_test  = sampled_data_test["isSignal"]

        sampled_data_train = sampled_data_train.drop("vM", axis=1)
        sampled_data_train = sampled_data_train.drop("vpCMS", axis=1)
        sampled_data_train = sampled_data_train.drop("__index__", axis=1)

        sampled_data_test  = sampled_data_test.drop("vM", axis=1)
        sampled_data_test  = sampled_data_test.drop("vpCMS",  axis=1)
        sampled_data_test  = sampled_data_test.drop("__index__",  axis=1)
        
        # column names list
        cols = list(sampled_data_train.columns)        
        # data scaling [0,1]
        sampled_data_train = pd.DataFrame(MinMaxScaler().fit_transform(sampled_data_train),columns = cols)
        sampled_data_test  = pd.DataFrame(MinMaxScaler().fit_transform(sampled_data_test), columns = cols)        

        if self.m_drop_class:
            X_train = sampled_data_train.drop("isSignal", axis=1)
            X_test  = sampled_data_test.drop("isSignal", axis=1)
        else:
            X_train = sampled_data_train
            X_test  = sampled_data_test

        return X_train, Y_train, X_test, Y_test
    
    # belle2 data preparation
    def belle2_3pi(self, data_train, data_test, sampling, sample_name):

        data_train = data_train.copy()
        data_train.loc[data_train["isSignal"] == 0, "isSignal"] = -1
        data_test = data_test.copy()
        data_test.loc[data_test["isSignal"] == 0, "isSignal"] = -1

        if(sampling or self.genetic): # sampling already done or not needed, don"t sample again!
            Y_train = data_train["isSignal"]
            Y_test  = data_test["isSignal"]
            # Data scaling [0,1]
            cols = list(data_train.columns)        
            data_train = pd.DataFrame(MinMaxScaler().fit_transform(data_train),columns = cols)
            data_train = data_train.drop("M", axis=1)
            data_train = data_train.drop("__index__", axis=1)

            data_test  = pd.DataFrame(MinMaxScaler().fit_transform(data_test) ,columns = cols)
            data_test  = data_test.drop("M", axis=1)
            data_test  = data_test.drop("__index__", axis=1)
            
            if self.m_drop_class:
                X_test  = data_test.drop("isSignal", axis=1)
                X_train = data_train.drop("isSignal", axis=1)
            else:
                X_test  = data_test
                X_train = data_train
            return X_train, Y_train, X_test, Y_test
        

        sampled_data_train = resample(data_train, replace = True, n_samples = 1000, random_state=None)
        sampled_data_test  = resample(data_test,  replace = False, n_samples = 10000, random_state=None)

        Y_train = sampled_data_train["isSignal"]
        Y_test  = sampled_data_test["isSignal"]

        if self.m_drop_class:
            sampled_data_train = sampled_data_train.drop("M", axis=1)
            sampled_data_test  = sampled_data_test.drop("M", axis=1)
        else:
            sampled_data_train = sampled_data_train
            sampled_data_test  = sampled_data_test

        # column names list
        cols = list(sampled_data_train.columns)        
        # data scaling [0,1]
        sampled_data_train = pd.DataFrame(MinMaxScaler().fit_transform(sampled_data_train),columns = cols)
        sampled_data_test  = pd.DataFrame(MinMaxScaler().fit_transform(sampled_data_test), columns = cols)
        
        X_train = sampled_data_train.drop("isSignal", axis=1)
        X_test  = sampled_data_test.drop("isSignal", axis=1)
        return X_train, Y_train, X_test, Y_test


    def belle2_challenge(self, data_train, data_test, sampling, sample_name):

        data_train = data_train.copy()
        data_train.loc[data_train["isSignal"] == 0, "isSignal"] = -1
        data_test = data_test.copy()
        data_test.loc[data_test["isSignal"] == 0, "isSignal"] = -1

        if(sampling or self.genetic and False): # sampling already done or not needed
            Y_train = data_train["isSignal"]
            Y_test  = data_test["isSignal"]
            # Data scaling [0,1]
            cols = list(data_train.columns)        
            data_train = pd.DataFrame(MinMaxScaler().fit_transform(data_train),columns = cols)
            data_train = data_train.drop("vM", axis=1)
            data_train = data_train.drop("vpCMS", axis=1)
            data_train = data_train.drop("__index__", axis=1)

            data_test  = pd.DataFrame(MinMaxScaler().fit_transform(data_test) ,columns = cols)
            data_test  = data_test.drop("vM", axis=1)
            data_test  = data_test.drop("vpCMS", axis=1)
            data_test  = data_test.drop("__index__", axis=1)

            if self.m_drop_class:            
                X_test  = data_test.drop("isSignal", axis=1)
                X_train = data_train.drop("isSignal", axis=1)
            else:
                X_test  = data_test
                X_train = data_train
            return X_train, Y_train, X_test, Y_test
        
        sampled_data_train = resample(data_train, replace = False, n_samples = 1000, random_state=None)
        sampled_data_test  = resample(data_test,  replace = False, n_samples = 10000, random_state=None)

        Y_train = sampled_data_train["isSignal"]
        Y_test  = sampled_data_test["isSignal"]

        sampled_data_train = sampled_data_train.drop("vM", axis=1)
        sampled_data_train = sampled_data_train.drop("vpCMS", axis=1)
        sampled_data_train = sampled_data_train.drop("__index__", axis=1)

        sampled_data_test  = sampled_data_test.drop("vM", axis=1)
        sampled_data_test  = sampled_data_test.drop("vpCMS",  axis=1)
        sampled_data_test  = sampled_data_test.drop("__index__",  axis=1)
        
        # column names list
        cols = list(sampled_data_train.columns)        
        # data scaling [0,1]
        sampled_data_train = pd.DataFrame(MinMaxScaler().fit_transform(sampled_data_train),columns = cols)
        sampled_data_test  = pd.DataFrame(MinMaxScaler().fit_transform(sampled_data_test), columns = cols)

        if self.m_drop_class:        
            X_train = sampled_data_train.drop("isSignal", axis=1)
            X_test  = sampled_data_test.drop("isSignal", axis=1)
        else:
            X_train = sampled_data_train
            X_test  = sampled_data_test
        return X_train, Y_train, X_test, Y_test


    #Titanic data preparation (for testing)
    def titanic(self, data_set):
        #return data_set #(for tmva prep)
        data_set = data_set.copy()        
        # set titles, these are quite problematic, so we format using the old way
        data_set.loc[:,"Title"] = data_set.Name.str.extract("([A-Za-z]+)", expand=False)
        data_set = data_set.drop(["Name"], axis=1)
        #change names
        data_set["Title"] = data_set["Title"].replace(["Lady", "Countess","Capt", "Col","Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"], "Rare")
        data_set["Title"] = data_set["Title"].replace("Mlle", "Miss")
        data_set["Title"] = data_set["Title"].replace("Ms", "Miss")
        data_set["Title"] = data_set["Title"].replace("Mme", "Mrs")
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        data_set["Title"] = data_set["Title"].map(title_mapping)
        data_set["Title"] = data_set["Title"].fillna(0)        

        #transform sex
        data_set.loc[data_set["Sex"]=="female", "Sex"]  = 0
        data_set.loc[data_set["Sex"]=="male",   "Sex"]  = 1

        #group/transforming ages
        data_set.loc[ data_set["Age"] <= 16, "Age"] = 0
        data_set.loc[(data_set["Age"] > 16) & (data_set["Age"] <= 32), "Age"] = 1
        data_set.loc[(data_set["Age"] > 32) & (data_set["Age"] <= 48), "Age"] = 2
        data_set.loc[(data_set["Age"] > 48) & (data_set["Age"] <= 64), "Age"] = 3
        data_set.loc[ data_set["Age"] > 64, "Age"] = 4

        #combine and drop features
        data_set["FamilySize"] = data_set["Siblings/Spouses Aboard"] + data_set["Parents/Children Aboard"] + 1
        data_set = data_set.drop(["Siblings/Spouses Aboard"], axis=1)
        data_set = data_set.drop(["Parents/Children Aboard"], axis=1)

        #create a new feature(s)
        data_set["IsAlone"] = 0
        data_set.loc[data_set["FamilySize"] == 1, "IsAlone"] = 1

        # Data scaling [0,1]
        data_set = pd.DataFrame(MinMaxScaler().fit_transform(data_set),
                                columns = list(data_set.columns))

        #change names
        title_mapping = {0: 1, 1: -1}
        #data_set["Survived"] = data_set["Survived"].map(title_mapping)
        #data_set["Survived"] = data_set["Survived"].fillna(0)
        data_set.loc[data_set["Survived"] == 0, "Survived"] = -1
        Y = data_set["Survived"]

        if self.m_drop_class: 
            X = data_set.drop("Survived", axis=1)
        else:
            X = data_set
        return X,Y
