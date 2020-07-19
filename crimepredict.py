#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import tree 
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score,recall_score,accuracy_score
import graphviz
import warnings
warnings.filterwarnings('ignore') #ignore warnings

#(1)讀取資料
data = pd.read_csv('train.csv',parse_dates=['Dates'])
data = data[['Dates','DayOfWeek','PdDistrict','X','Y','Category']]

#data.info()
#(2-1)觀察資料發現沒有空值

#(2-2)把DayOfWeek,PdDistrict轉為dummy variables
dayofweek_dummies = pd.get_dummies(data['DayOfWeek'])
data = data.join(dayofweek_dummies)
data.drop(['DayOfWeek'],axis=1,inplace=True)
pddistrict_dummies = pd.get_dummies(data['PdDistrict'])
data = data.join(pddistrict_dummies)
data.drop(['PdDistrict'],axis=1,inplace=True)

#(2-3)從Ｄates抓取小時轉為dummy variables
hour = data['Dates'].dt.hour
hour_dummies = pd.get_dummies(hour)
data = data.join(hour_dummies)
data.drop(['Dates'],axis=1,inplace=True)

#(2-4)對X跟Y進行處理：把Ｘ和Ｙ資料標準化
xy_scaler = StandardScaler()
data[['X','Y']] = xy_scaler.fit_transform(data[['X','Y']])

#(2-4)對X跟Y進行處理：透過PCA進行降維
#pca = PCA(n_components=1)
#data['Coordinates'] = pca.fit_transform(data[['X','Y']])
#data.drop(data[['X','Y']],axis=1,inplace=True)

#(2-5)把資料隨機拆成training set(75%)跟test set(25%)
train,test = train_test＿split(data,test_size=0.25)
#random＿state沒設定就是使用預設的np.random

#(3)使用DecisionTreeClassifier進行預測
#train_y = train['Category']
#train.drop(['Category'],axis=1,inplace=True)
#train_x = train
#test_y = test['Category']
#test.drop(['Category'],axis=1,inplace=True)
#test_x = test
#clf = tree.DecisionTreeClassifier(max_depth=3)
#clf.fit(train_x,train_y)
#pred_y = clf.predict(test)
predictors = ['X','Y','Friday','Monday','Saturday','Sunday','Thursday','Tuesday',
              'Wednesday','BAYVIEW','CENTRAL','INGLESIDE','MISSION','NORTHERN',
              'PARK','RICHMOND','SOUTHERN','TARAVAL','TENDERLOIN',0,1,2,3,4,5,6,
              7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
clf = tree.DecisionTreeClassifier(max_depth=3,presort=True)
#presort the data to speed up the finding of best splits in fitting
clf.fit(train[predictors],train['Category'])
pred_y = clf.predict(test[predictors])

#(4)列印Precision,Recall,F1-score,Accuracy資料
#print(classification_report(test['Category'],pred_y))
print('precision  ',precision_score(test['Category'],pred_y,average='weighted'))
print('recall  ',recall_score(test['Category'],pred_y,average='weighted'))
#macro:計算二分類metrics的均值，為每個類给出相同權重的分值，無法區分出某些小類重要性
#micro:给出了每個樣本類以及它對個metrics的貢獻的pair(sample weight)，大類將被忽略
#weighted:對於不均衡数量的類来说，計算二分類metrics的平均，通過在每個類的score上進行加權
print('accuracy  ',accuracy_score(test['Category'],pred_y))

#(5)產出決策樹
#dot_data = tree.export_graphviz(clf,feature_names=predictors,out_file=None) 
#graph = graphviz.Source(dot_data) 
#graph.render("crime") 