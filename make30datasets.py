import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# from ucimlrepo import fetch_ucirepo

import csv
# DATASETS_DIR = ('datasets/')
DATASETS_DIR = ('datasets/regressionDatasets/datasets-regression/')

import scipy.io
def read_dataset_mat(which):
    file = scipy.io.loadmat(DATASETS_DIR + which+'.mat')
    data=file['trainD']
    trainD=make_dataframe(data)
    file2=scipy.io.loadmat(DATASETS_DIR + which+'Test'+'.mat')
    data2 = file2['testD']
    testD = make_dataframe(data2)
    df= pd.concat([trainD,testD], ignore_index=True, sort=False)
    x = df.iloc[:, :-1]
    y = df['Y']
    le = LabelEncoder()
    yle1 = le.fit_transform(y)
    yle = pd.Series(yle1)
    return df, x, y, yle1, yle
def make_dataframe(data):
    T = []

    for i in range(len(data[0])-1):
        string1 = 'X_' + str(i)
        T.append(string1)

    T.append('Y')

    df = pd.DataFrame(data=data, columns=T)
    return df


def read_dataset_csv(which):
    df = pd.read_csv(DATASETS_DIR + which + '.csv')

    # remove ID column for real estate dataset
    if which.lower() == "real_estate_valuation":
        df = df.iloc[:, 1:]   # drop first column (No)

    x = df.iloc[:, :-1]
    y = df['Y']

    le = LabelEncoder()
    yle1 = le.fit_transform(y)
    yle = pd.Series(yle1)

    return df, x, y, yle1, yle

def read_dataset_csv2(which):
    df = pd.read_csv(DATASETS_DIR + which + '.csv')
    z=df.shape

    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    le = LabelEncoder()
    yle1 = le.fit_transform(y)
    yle = pd.Series(yle1)
    return df, x, y, yle1, yle
def read_dataset_dat(which):
    with open(DATASETS_DIR + which + '.dat', 'r') as f:
        content = f.readlines()
    cols = []
    i = 0
    for line in content:
        if line.startswith("@outputs"):
            index = line.split('\n')[0]
            className = index.split(' ')[-1]

    for line in content:
        if line.startswith("@attribute " + className):
            cols.append('Y')
        elif line.startswith("@attribute"):
            cols.append('X' + str(i))
            i = i + 1

    df = pd.read_csv(DATASETS_DIR + which + '.dat', delimiter=',', comment='@', names=cols)
    x = df.iloc[:, :-1]
    y = df['Y']
    le = LabelEncoder()
    yle1 = le.fit_transform(y)
    yle = pd.Series(yle1)
    return df, x, y, yle1, yle

# def readurl(name):
#     X = []
#     # fetch dataset
#     if name == 'balance_scale':
#         id1 = 12
#     elif name == 'iris':
#         id1 = 53
#     elif name == 'wine':
#         id1 = 109
#
#     df = fetch_ucirepo(id=id1)
#     X = df.data.features
#     y = df.data.targets
#
#     le = LabelEncoder()
#     yle1 = le.fit_transform(y)
#     yle = pd.Series(yle1)
#     return df, X, y, yle1, yle

# datasetName = ['yeast', 'wav', 'movement_libras', 'vowel', 'mcd3', 'mcd10', 'segment','bupa', 'haberman', 'ionosphere', 'pimaindiansdiabetes', 'wdbc', 'spambase']
# datasetName = ["sonar", "thyroid1", 'bupa', 'haberman', 'ionosphere', 'pimaindiansdiabetes', 'wdbc', 'spambase']
#
# datasetName=['balance_scale', 'iris', 'wine']
datasetName = ['Real_estate_valuation']
# datasetName=['bm1','bm4','RatPol2D','Pollen','keijzer5','Concrete','Toxicity','X05Y1Z15']
#datasetName = ['Pollen']
#datasetName = ['heart', 'yeast', 'wav', 'movement_libras', 'vowel', 'segment', 'mcd10', 'mcd3']
#datasetName = ['banana1','breast_cancer1','diabetis1','flare_solar1','german1','image1','ringnorm1','splice1','thyroid1','titanic1','twonorm1','waveform1']
#datasetName =['banana1','breast_cancer1','diabetis1','flare_solar1','german1','image1','ringnorm1','splice1','thyroid1','titanic1','twonorm1','waveform1','heart','ionosphere', 'pimaindiansdiabetes','wdbc','spambase','banana1', 'breast_cancer1', 'diabetis1', 'flare_solar1', 'german1', 'image1', 'ringnorm1','splice1', 'thyroid1', 'titanic1', 'twonorm1', 'waveform1', 'movement_libras', 'yeast', 'wav', 'movement_libras', 'vowel', 'segment']
def split_stratified_into_train_val_test(x, y,frac_train=0.6, frac_val=0.15, frac_test=0.25, random_state=None):

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))
    X = x
    y = y

    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          test_size=(1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)
    return df_train, df_test, df_val, y_train, y_test, y_val

# readurl('balance_scale')


for which in datasetName:

    print('dataset:' + which)
    if which in ('Real_estate_valuation','bioavailability','bm1','bm4','RatPol2D','Pollen','keijzer5','Concrete','Toxicity','X05Y1Z15','Concrete_Data','Admission_Predict_kaggle','banana1','breast_cancer1','diabetis1','flare_solar1','german1','image1','ringnorm1','splice1','thyroid1','titanic1','twonorm1','waveform1',"heart", "wav", "banana1", "vow", "mcd3", "mcd10", "parkinsons", "pimaindiansdiabetes"):
        df, x, y, yle1, yle = read_dataset_csv(which)
    elif which in ('Concrete','Pollen'):
        df, x, y, yle1, yle = read_dataset_mat(which)
    # elif which in ('balance_scale', 'iris', 'wine'):
    #     df, x, y, yle1, yle = readurl(which)
    else:
        df, x, y, yle1, yle = read_dataset_dat(which)

    for r in range(30):
        X_train, X_test, X_val, y_train, y_test,y_val=split_stratified_into_train_val_test(x, yle,
                                         frac_train=0.50, frac_val=0.20, frac_test=0.30,
                                         random_state=r)
        Y_tr = pd.DataFrame(y_train)
        Y_t = pd.DataFrame(y_test)
        Y_v = pd.DataFrame(y_val)
        X_train['Y'] = Y_tr[0]
        X_test['Y'] = Y_t[0]
        X_val['Y'] = Y_v[0]

        X_train.to_csv('datasetbreak2/' + which + str(r) + 'train.csv', index=False)
        X_test.to_csv('datasetbreak2/' + which + str(r) + 'test.csv', index=False)
        X_val.to_csv('datasetbreak2/' + which + str(r) + 'val.csv', index=False)
