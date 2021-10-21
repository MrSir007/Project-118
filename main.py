import csv
from numpy import dot
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn import metrics
from sklearn.tree import export_graphviz as eg
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus as pp

columnName = ["Prg","Glu","BP","Skin","Insu","BMI","DPF","Age","Out"]
getData = pd.read_csv("diabetes.csv",names=columnName).iloc[1:]
print(getData.head())

# To train and test
features = ["Prg","Glu","BP","Insu","BMI","DPF","Age"]
X = getData[features]
Y = getData.Out
xTrain, xTest, yTrain, yTest = tts(X,Y,test_size=0.3,random_state=1)
# To initialise the decision tree
dtc = DecisionTreeClassifier()
dtc = dtc.fit(xTrain, yTrain)
yPredict = dtc.predict(xTest)
print("Accuracy:", metrics.accuracy_score(yTest, yPredict))
# To store the data from "dtc" as text
dotdata = StringIO()
eg(dtc,out_file=dotdata,filled=True,rounded=True,special_characters=True,feature_names=features,class_names=["0","1"])
print(dotdata.getValue())

decisionTree = pp.graph_from_dot_data(dotdata.getValue())
decisionTree.write_png("diabetes.png")
Image(decisionTree.create_png())