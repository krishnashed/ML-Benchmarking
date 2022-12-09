import daal4py as d4p
import xgboost as xgb
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

def pd_read_csv(f, c=None, t=np.float64):
    return pd.read_csv(f, usecols=c, delimiter=',', header=None, dtype=t)

readcsv=pd_read_csv
# Path to data
train_file = "./data/df_classification_train.csv"
test_file = "./data/df_classification_test.csv"

# Data reading
X_train = readcsv(train_file, range(3), t=np.float32)
y_train = readcsv(train_file, range(3, 4), t=np.float32)
X_test = readcsv(test_file, range(3), t=np.float32)
y_test = readcsv(test_file, range(3, 4), t=np.float32)

# Datasets creation
xgb_train = xgb.DMatrix(X_train, label=np.array(y_train))
xgb_test = xgb.DMatrix(X_test, label=np.array(y_test))


# training parameters setting
params = {
    'max_bin': 256,
    'scale_pos_weight': 2,
    'lambda_l2': 1,
    'alpha': 0.9,
    'max_depth': 8,
    'num_leaves': 2**8,
    'verbosity': 0,
    'objective': 'multi:softmax',
    'learning_rate': 0.3,
    'num_class': 5,
}

# Training
xgb_model = xgb.train(params, xgb_train, num_boost_round=100)

# XGBoost prediction (for accuracy comparison)
t0 = time.time()
xgb_prediction = xgb_model.predict(xgb_test)
t1 = time.time()
xgb_errors_count = np.count_nonzero(xgb_prediction - np.ravel(y_test))

xgb_total = t1-t0
print(xgb_total)

# Conversion to daal4py
daal_model = d4p.get_gbt_model_from_xgboost(xgb_model)

# daal4py prediction
daal_predict_algo = d4p.gbt_classification_prediction(
    nClasses=params["num_class"],
    resultsToEvaluate="computeClassLabels",
    fptype='float'
)
t0 = time.time()
daal_prediction = daal_predict_algo.compute(X_test, daal_model)
t1 = time.time()
daal_errors_count = np.count_nonzero(daal_prediction.prediction -  y_test)

d4p_total = t1-t0
print(d4p_total)


assert np.absolute(xgb_errors_count - daal_errors_count) == 0
y_test = np.ravel(y_test)
daal_prediction = np.ravel(daal_prediction.prediction)

print("\nXGBoost prediction results (first 10 rows):\n", xgb_prediction[0:10])
print("\ndaal4py prediction results (first 10 rows):\n", daal_prediction[0:10])
print("\nGround truth (first 10 rows):\n", y_test[0:10])

print("XGBoost errors count:", xgb_errors_count)
print("XGBoost accuracy score:", 1 - xgb_errors_count / xgb_prediction.shape[0])

print("\ndaal4py errors count:", daal_errors_count)
print("daal4py accuracy score:", 1 - daal_errors_count / daal_prediction.shape[0])

print("\n XGBoost Prediction Time:", xgb_total)
print("\n daal4py Prediction Time:", d4p_total)
print("\nAll looks good!")

# Performance
left = [1,2]
pred_times = [xgb_total, d4p_total]
tick_label = ['XGBoost Prediction', 'daal4py Prediction']
plt.bar(left, pred_times, tick_label = tick_label, width = 0.5, color = ['red', 'blue'])
plt.xlabel('Prediction Method'); plt.ylabel('time,s'); plt.title('Prediction time,s')
plt.show()
print("speedup:",xgb_total/d4p_total)

# Accuracy
left = [1,2]
xgb_acc = 1 - xgb_errors_count / xgb_prediction.shape[0]
d4p_acc = 1 - daal_errors_count / daal_prediction.shape[0]
pred_acc = [xgb_acc, d4p_acc]
tick_label = ['XGBoost Prediction', 'daal4py Prediction']
plt.bar(left, pred_acc, tick_label = tick_label, width = 0.5, color = ['red', 'blue'])
plt.xlabel('Prediction Method'); plt.ylabel('accuracy, %'); plt.title('Prediction Accuracy, %')
plt.show()
print("Accuracy Difference",xgb_acc-d4p_acc)