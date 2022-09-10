import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from google.colab import drive
drive.mount('/content/drive')

dc = pd.read_excel(url_contaminantes)

dm = pd.read_excel(url_mediciones)

ds = pd.read_excel(url_datos)
ds.head()

dc.shape

dm.shape

dm.head()

contaminantes = dc.shape[0]
dc = dc.merge(dm, how = 'inner', on = ['Fecha'])
dc.head()

dc.columns

dc.shape

dc.dtypes

dc.isna().sum()

"""#Relleno de datos faltantes"""

plt.figure(figsize = (23, 17))
mask = np.triu(np.ones_like(dc.corr(), dtype=bool))
sns.heatmap(dc.corr(), annot = False, cmap="crest", mask=mask)
plt.show()

cols = ['SO2_SE','CO_SE','O3_SE','PM10_SE','PM2.5_SE','TOUT_SE','RH_SE','PRS_SE','WS_SE','WS_SE2','WD_SE2']
df = dc[cols]
df.head()

plt.figure(figsize = (15, 9))
sns.heatmap(df.corr(), annot=True, cmap="magma")
plt.show()

"""#Normalizacion de la base de datos"""

plt.figure(figsize = (15, 9))
sns.histplot(data=df)
plt.show()

#Normalizacion de la base de datos
dfn = (df-np.min(df))/(np.max(df)-np.min(df))
dfn.head()

plt.figure(figsize = (15, 9))
sns.histplot(data=dfn)
plt.show()

print(dfn.isna().sum())

"""#Quitar outliers"""

plt.figure(figsize=(10,6)) 
dfn.boxplot()

dfn.shape

for i in range(0, dfn.shape[1]):
  Q1w = np.percentile(dfn.iloc[:,i],25)
  Q3w = np.percentile(dfn.iloc[:,i],75)
  iqr = Q3w - Q1w
  ls = Q3w + 1.5*iqr
  li = Q1w - 1.5*iqr
  lsup = dfn[dfn.iloc[:,i]>ls]
  linf = dfn[dfn.iloc[:,i]<li]
  dfn.drop(lsup.index,inplace=True)
  dfn.drop(linf.index,inplace=True)

plt.figure(figsize=(10,6)) 
dfn.boxplot()

dfn.shape

plt.figure(figsize = (15, 9))
sns.heatmap(dfn.corr(), annot=True, cmap="magma")
plt.show()

"""#Agregar variable de clasificacion"""

ds.drop(columns = ['Unnamed: 0','CO_SE','O3_SE','SO2_SE','PM10_SE','PM2.5_SE','TOUT_SE','RH_SE','SR_SE','PRS_SE','WS_SE'], inplace=True)
dfn['Fecha'] = dc['Fecha']
dfn.head()

from datetime import datetime, date, timedelta

start_date = date(2014, 1, 31)
end_date = date(2022, 8, 15)
delta = timedelta(days = 1)
sequia = 1
df_sequia = pd.DataFrame()

while start_date < end_date:
  start_hour = datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0)
  end_hour = start_hour + delta
  while start_hour < end_hour:
    if pd.Timestamp(start_hour) in ds['Fecha'].tolist():
      sequia = ds[ds['Fecha'] == pd.Timestamp(start_hour)]['Sequia'].item()
    else:
      df_sequia = df_sequia.append({'Fecha': pd.Timestamp(start_hour), 'Sequia': sequia}, ignore_index = True)
    start_hour += timedelta(hours = 1)
  start_date += delta

df_sequia.head()

dfn = dfn.merge(df_sequia, how = 'inner', on = ['Fecha'])
dfn['Fecha'].head()

dfn.head()

dfn.shape

"""#Analisis"""

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

target = dfn['Sequia']
dfi = dfn.copy()
dfn.drop(columns=['Sequia','Fecha'], inplace=True)

#Train test split
X_train,X_test,y_train,y_test= train_test_split(dfn,target,test_size=0.25, random_state=0)

#Function to print statistics
def stats(test,prediction):
  matrix = confusion_matrix(test, prediction)
  print(matrix)
  print('accuracy:',accuracy_score(test,prediction))

#Support Vector Classification linear
clf = SVC(kernel='linear')
clf.fit(X_train,y_train)
predict = clf.predict(X_test)
print(predict)
stats(y_test,predict)

#Support Vector Classification rbf
clf2 = SVC(kernel='rbf')
clf2.fit(X_train,y_train)
predict2 = clf2.predict(X_test)
print(predict2)
stats(y_test,predict2)

#Knn
clf3 = KNeighborsClassifier(n_neighbors=3)
clf3.fit(X_train, y_train)
predict3 = clf3.predict(X_test)
print(predict3)
stats(y_test,predict3)

#Decision tree
clf4 = DecisionTreeClassifier(random_state=0)
clf4.fit(X_train, y_train)
predict4 = clf4.predict(X_test)
print(predict4)
stats(y_test,predict4)

#Random forest
clf5 = RandomForestClassifier()
clf5.fit(X_train, y_train)
predict5 = clf5.predict(X_test)
print(predict5)
stats(y_test,predict5)

#Naive Bayes
clf6 = GaussianNB()
clf6.fit(X_train, y_train)
predict6 = clf6.predict(X_test)
print(predict6)
stats(y_test,predict6)

#Linear Discriminant Analysis
clf7 = LinearDiscriminantAnalysis()
clf7.fit(X_train, y_train)
predict7 = clf7.predict(X_test)
print(predict7)
stats(y_test,predict7)

x = ['SVCL','SVCR','KNN','DT','RF','BN','LDA']
y = [accuracy_score(y_test,predict), accuracy_score(y_test,predict2), accuracy_score(y_test,predict3)
, accuracy_score(y_test,predict4), accuracy_score(y_test,predict5), accuracy_score(y_test,predict6), accuracy_score(y_test,predict7)]
plt.figure(figsize=(10,6)) 
sns.barplot(x,y)
