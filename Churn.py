import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
import csv
from sklearn import preprocessing

df = pd.read_csv("train.csv")
df1 = pd.read_csv("test.csv")

X = pd.DataFrame()
X = df

%matplotlib inline
import matplotlib.pyplot as plt
num_bins = 10
X.hist(bins=num_bins, figsize=(20,15))
plt.savefig("hr_histogram_plots")
plt.show()

cat_vars=["COLLEGE","REPORTED_SATISFACTION","REPORTED_USAGE_LEVEL","CONSIDERING_CHANGE_OF_PLAN"]
for var in cat_vars:
	cat_list='var'+'_'+var
	cat_list = pd.get_dummies(X[var], prefix=var)
	X1=X.join(cat_list)
	X=X1
	
X.dropna(axis=0)
y = X['LEAVE']
X = X.drop(["LEAVE","REPORTED_SATISFACTION","REPORTED_USAGE_LEVEL","CONSIDERING_CHANGE_OF_PLAN"], axis=1)
X = pd.get_dummies(X, ["COLLEGE"])

testset = pd.DataFrame()
testset = df1
testset = testset.drop(["REPORTED_SATISFACTION","REPORTED_USAGE_LEVEL","CONSIDERING_CHANGE_OF_PLAN"], axis=1)
testset = pd.get_dummies(testset, ["COLLEGE"])

min_samples_leaf = 0.007
max_depth = 6
model = tree.DecisionTreeClassifier(criterion="entropy",min_samples_leaf = min_samples_leaf, max_depth = max_depth)
model.fit(X, y)
pred = model.predict(testset)
print "Sum of 1s", sum(pred)
