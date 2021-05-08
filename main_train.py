import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

dict_emp_len ={'\+ years':'',' years':'',' year':'','< 1':'0','n/a':'0'}
dict_last_week_pay = {'th week':'','NA':''}
dict_sub_grade = {'A':'0','B':'1','C':'2','D':'3','E':'4','F':'5','G':'6'}

df_train=pd.read_csv("train_indessa.csv")
df_train['term'].replace(to_replace=' months', value='', regex=True, inplace=True)
df_train['term'] = pd.to_numeric(df_train['term'], errors='raise')

df_train['emp_length'].replace(to_replace=dict_emp_len, value=None, regex=True, inplace=True)
df_train['emp_length'] = pd.to_numeric(df_train['emp_length'], errors='raise')

df_train['last_week_pay'].replace(to_replace=dict_last_week_pay, value=None, regex=True, inplace=True)
df_train['last_week_pay'] = pd.to_numeric(df_train['last_week_pay'], errors='raise')

df_train['sub_grade'].replace(to_replace=dict_sub_grade, value=None, regex=True, inplace=True)
df_train['sub_grade'] = pd.to_numeric(df_train['sub_grade'], errors='raise')


df_test=pd.read_csv("test_indessa.csv")
df_test['term'].replace(to_replace=' months', value='', regex=True, inplace=True)
df_test['term'] = pd.to_numeric(df_test['term'], errors='raise')

df_test['emp_length'].replace(to_replace=dict_emp_len, value=None, regex=True, inplace=True)
df_test['emp_length'] = pd.to_numeric(df_test['emp_length'], errors='raise')

df_test['last_week_pay'].replace(to_replace=dict_last_week_pay, value=None, regex=True, inplace=True)
df_test['last_week_pay'] = pd.to_numeric(df_test['last_week_pay'], errors='raise')

df_test['sub_grade'].replace(to_replace=dict_sub_grade, value=None, regex=True, inplace=True)
df_test['sub_grade'] = pd.to_numeric(df_test['sub_grade'], errors='raise')

df_train_col = list(df_train)
for col in df_train_col:
    print('Imputation with Median: %s' % (col))
print("done")

cols = ['term', 'loan_amnt', 'funded_amnt', 'last_week_pay', 'int_rate', 'sub_grade', 'annual_inc', 'dti', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'revol_bal', 'revol_util', 'total_acc', 'total_rec_int', 'mths_since_last_major_derog', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim', 'emp_length']

for col in cols:
    df_train[col].fillna(df_train[col].median(), inplace=True)

cols = ['acc_now_delinq', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'collections_12_mths_ex_med']
for col in cols:
    print('Imputation with Zero: %s' % (col))
    df_train[col].fillna(0, inplace=True)

train_target = pd.DataFrame(df_train['loan_status'])

selected_cols = ['member_id', 'emp_length', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'sub_grade', 'int_rate', 'annual_inc', 'dti', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'revol_bal', 'revol_util', 'total_acc', 'total_rec_int', 'total_rec_late_fee', 'mths_since_last_major_derog', 'last_week_pay', 'tot_cur_bal', 'total_rev_hi_lim', 'tot_coll_amt', 'recoveries', 'collection_recovery_fee', 'term', 'acc_now_delinq', 'collections_12_mths_ex_med']

total_train = df_train[selected_cols]
total_test = df_test[selected_cols]

test_member_id = pd.DataFrame(df_test['member_id'])

X_train, X_test, y_train, y_test = train_test_split(np.array(total_train), np.array(train_target), test_size=0.20)
eval_set=[(X_test, y_test)]

clf = xgboost.sklearn.XGBClassifier( objective="binary:logistic", learning_rate=0.001, seed=9616, max_depth=20, gamma=10, n_estimators=2)

clf.fit(X_train, y_train, early_stopping_rounds=20, eval_metric="auc", eval_set=eval_set, verbose=True)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(np.array(y_test).flatten(), y_pred)
print("Accuracy: %.10f%%" % (accuracy * 100.0))

accuracy_per_roc_auc = roc_auc_score(np.array(y_test).flatten(), y_pred)
print("ROC-AUC: %.10f%%" % (accuracy_per_roc_auc * 100))

