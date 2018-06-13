from Adaboost import *
from Plot2D import *
import pandas as pd

#data.csv is created by make_data.py
data=pd.read_csv('data.csv')

#get X and y
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

#train the AdaboostClassifier
clf=AdaboostClassifier()
times=clf.fit(X,y)

#plot original data
Plot2D(data).pause(3)

#plot Adaboost decision_threshold
for i in xrange(times):
	if clf.weak[i].decision_feature==0:
		plt.plot([clf.weak[i].decision_threshold,clf.weak[i].decision_threshold],[0,8])
	else:
		plt.plot([0,8],[clf.weak[i].decision_threshold,clf.weak[i].decision_threshold])
plt.pause(3)
