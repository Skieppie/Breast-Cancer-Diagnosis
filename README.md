# Breast-Cancer-Diagnosis\
\
\
\
Manual breast cancer diagnosis by doctoes is a slow, laboring tasks, prone to errors. By using computer algorithms, the accuracy of the diagnonis can reach even 100%. Because of the efficiency of algorithmical detection, doctors in Thailand went on strike to protest the implementation of such an algorithm as they would lose an integral part of their job.\
\
The following algorithm uses XGBoost  and it has an accouracy of 97%.\
\
\
\
import numpy as np\
import pandas as pd\
import matplotlib.pyplot as plt\
import seaborn as sns\
import xgboost\
import os\
\
for dirname, _, filenames in os.walk('/kaggle/input'):\
    for filename in filenames:\
        print(os.path.join(dirname, filename))\
        \
train_df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')\
train_df\
\
x_train = train_df.drop(['Unnamed: 32', 'id', 'diagnosis'], axis=1)\
y_train = train_df.diagnosis\
\
print("Features", x_train)\
print("Labels", y_train)\
\
B, M = y_train.value_counts()\
\
print('Number of Benign: ' , B)\
print('Number of Malignant : ' , M)\
\
f,ax = plt.subplots(figsize=(18, 18))\
corr_mat = sns.heatmap(x_train.corr(), cmap='coolwarm', annot=False, linewidths=.5, fmt= '.1f',ax=ax)\
\
from sklearn.svm import SVC\
from sklearn.model_selection import StratifiedKFold\
from sklearn.feature_selection import RFECV\
\
svc = SVC(kernel='linear')\
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2), scoring='accuracy')\
rfecv.fit(x_train.values, y_train.values)\
\
print("Optimal number of features : %d" % rfecv.n_features_)\
best_features = list(x_train.columns[rfecv.support_])\
print('Best features :', best_features)\
\
f,ax = plt.subplots(figsize=(12, 12))\
corr_mat = sns.heatmap(x_train[best_features].corr(), cmap='coolwarm', annot=False, linewidths=.5, fmt= '.1f',ax=ax)\
\
from sklearn.preprocessing import StandardScaler\
from sklearn.model_selection import train_test_split\
from sklearn.metrics import accuracy_score, confusion_matrix\
\
def preprocess(X):\
    #extracting chosen features\
    X = X[best_features]\
    #Standardizing Features\
    sc = StandardScaler()\
    X = sc.fit_transform(X)\
    return X\
    \
x_train = preprocess(x_train)\
x_train\
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.20, random_state=42, shuffle=False)\
model = xgboost.XGBClassifier()\
model.fit(x_train, y_train)\
prediction = model.predict(x_test)\
\
accuracy = accuracy_score(y_test, prediction) * 100\
confusion = confusion_matrix(y_test, prediction)\
precision = confusion[0][0]/(confusion[0][0] + confusion[1][0]) * 100\
xg_recall1 = confusion[0][0]/(confusion[0][0] + confusion[0][1]) * 100\
score = ((2 * precision * xg_recall1) / (precision + xg_recall1)) / 100\
\
print("Accuracy:", accuracy)\
print("Precision:", precision)\
print("Score:", score)\
\
------------
OUT:\
![out1](https://user-images.githubusercontent.com/99653642/187438014-42386d69-b5a9-4fe5-bce7-210db8800c20.png)
\
Number of Benign:  357\
Number of Malignant :  212\
\
![__results___7_0](https://user-images.githubusercontent.com/99653642/187438353-020414c8-2d77-4474-9cd2-5306771fa5ed.png)
\
Optimal number of features : 17\
\
![__results___10_0](https://user-images.githubusercontent.com/99653642/187438473-fcd35a37-0ba8-4fc9-9500-1428bd1fc0b5.png)
\
array([[ 1.09706398,  1.26993369,  1.56846633, ...,  2.29607613,\
         2.75062224,  1.93701461],\
       [ 1.82982061,  1.68595471, -0.82696245, ...,  1.0870843 ,\
        -0.24388967,  0.28118999],\
       [ 1.57988811,  1.56650313,  0.94221044, ...,  1.95500035,\
         1.152255  ,  0.20139121],\
       ...,\
       [ 0.70228425,  0.67267578, -0.84048388, ...,  0.41406869,\
        -1.10454895, -0.31840916],\
       [ 1.83834103,  1.98252415,  1.52576706, ...,  2.28998549,\
         1.91908301,  2.21963528],\
       [-1.80840125, -1.81438851, -3.11208479, ..., -1.74506282,\
        -0.04813821, -0.75120669]])\
        \
