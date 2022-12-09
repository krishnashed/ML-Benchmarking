import numpy as np
import os
import matplotlib.pyplot as plt
import requests
import pandas as pd
import sys
import xgboost as xgb
import time

def load_higgs(nrows_train, nrows_test, dtype=np.float32):
    if not os.path.isfile("./HIGGS.csv.gz"):
        print("Loading data set...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
        myfile = requests.get(url)
        with open('./HIGGS.csv.gz', 'wb') as f:
            f.write(myfile.content)
    print("Reading data set...")
    data = pd.read_csv("./HIGGS.csv.gz", delimiter=",", header=None, compression="gzip", dtype=dtype, nrows=nrows_train+nrows_test)
    print("Pre-processing data set...")
    data = data[list(data.columns[1:])+list(data.columns[0:1])]
    n_features = data.shape[1]-1
    train_data = np.ascontiguousarray(data.values[:nrows_train,:n_features])
    train_label = np.ascontiguousarray(data.values[:nrows_train,n_features])
    test_data = np.ascontiguousarray(data.values[nrows_train:nrows_train+nrows_test,:n_features])
    test_label = np.ascontiguousarray(data.values[nrows_train:nrows_train+nrows_test,n_features])
    n_classes = len(np.unique(train_label))
    print(sys.getsizeof(train_data))
    return train_data, train_label, test_data, test_label, n_classes, n_features

train_data, train_label, test_data, test_label, n_classes, n_features = load_higgs(10000, 10000)

# Set XGBoost parameters
xgb_params = {
    'verbosity':                    0,
    'alpha':                        0.9,
    'max_bin':                      256,
    'scale_pos_weight':             2,
    'learning_rate':                0.1,
    'subsample':                    1,
    'reg_lambda':                   1,
    "min_child_weight":             0,
    'max_depth':                    8,
    'max_leaves':                   2**8,
    'objective':                    'binary:logistic',
    'predictor':                    'cpu_predictor',
    'tree_method':                  'hist',
    'n_estimators':                1000
}

# Train the model
t0 = time.time() #begin timer
model_xgb= xgb.XGBClassifier(**xgb_params)
model_xgb.fit(train_data, train_label)
t1 = time.time() #end timer

#predict label using test data
result_predict_xgb_test = model_xgb.predict(test_data)

# Check model accuracy
acc = np.mean(test_label == result_predict_xgb_test)
print(acc)

xgb_total = t1-t0
print(xgb_total)

filename = "./perf_numbers.csv"

xgb_ver= xgb.__version__

if not os.path.isfile(filename):
    df = pd.DataFrame([[xgb_ver,xgb_total]], columns = ["XGBoost Version",  "Time in Sec"])
    df.to_csv(filename, index=False) 
else:
    df = pd.read_csv(filename)
    if not df.shape[0]==2:
        df2 = pd.DataFrame([[xgb_ver,xgb_total]], columns = ["XGBoost Version",  "Time in Sec"])
        df = df.append(df2, ignore_index=True)
        df.to_csv(filename, index=False)


if ((os.path.isfile(filename)) and (df.shape[0]==2)):
    df.plot(x='XGBoost Version', y='Time in Sec', kind='bar',width = 0.5)
    plt.xlabel('XGBoost Version'); plt.ylabel('Time in Sec'); plt.title('XGBoost Performance Comparison')
    plt.show()

print(df)