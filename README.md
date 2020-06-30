import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


import os
for dirname, _, filenames in os.walk('/teju/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        place = pd.read_csv('/teju/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
        
        print(place.shape)
place.head(20)
print(place.isna().any())
#place = place.set_index('sl_no')
place.head()

print(place.describe())

pv1 = place.pivot_table(index = 'gender', columns = 'status',values = 'ssc_p' )

x = np.arange(len(pv1.index))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize = (5,5))
rects1 = ax.bar(x - width/2, pv1['Not Placed'] , width, label='Not Placed')
rects2 = ax.bar(x + width/2, pv1['Placed'] , width, label='Placed')

ax.spines["top"].set_color("None")
ax.spines["right"].set_color("None")
ax.set_ylabel('Scores')
ax.set_title('Average Senior secondary percentage by Gender')
ax.set_xticks(x)
ax.set_ylim(0,100)
ax.set_xticklabels(pv1.index)
ax.legend()
plt.show()

pv2 = place.pivot_table(index = 'gender', columns = 'status',values = 'etest_p' )

x = np.arange(len(pv2.index))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize = (5,5))
rects1 = ax.bar(x - width/2, pv2['Not Placed'] , width, label='Not Placed')
rects2 = ax.bar(x + width/2, pv2['Placed'] , width, label='Placed')

ax.spines["top"].set_color("None")
ax.spines["right"].set_color("None")
ax.set_ylabel('Scores')
ax.set_title('Average Employability test scores by Gender')
ax.set_xticks(x)
ax.set_ylim(0,100)
ax.set_xticklabels(pv1.index)
ax.legend()
plt.show()
pv3 = place.pivot_table(index = 'gender', columns = 'status',values = 'mba_p' )

x = np.arange(len(pv3.index))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize = (5,5))
rects1 = ax.bar(x - width/2, pv3['Not Placed'] , width, label='Not Placed')
rects2 = ax.bar(x + width/2, pv3['Placed'] , width, label='Placed')

ax.spines["top"].set_color("None")
ax.spines["right"].set_color("None")
ax.set_ylabel('Scores')
ax.set_title('Average MBA percentage by Gender')
ax.set_xticks(x)
ax.set_ylim(0,100)
ax.set_xticklabels(pv1.index)
ax.legend()
plt.show()
pv4 = place.pivot_table(index = 'degree_t', columns = 'status', values = 'gender', aggfunc = 'count')
print(pv4)
x = np.arange(len(pv4.index))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize = (5,5))
rects1 = ax.bar(x - width/2, pv4['Not Placed'] , width, label='Not Placed')
rects2 = ax.bar(x + width/2, pv4['Placed'] , width, label='Placed')

ax.spines["top"].set_color("None")
ax.spines["right"].set_color("None")
ax.set_ylabel('Count')
ax.set_title('Count of placement status of students by degree')
ax.set_xticks(x)
ax.set_ylim(0,110)
ax.set_xticklabels(pv4.index)
ax.legend()
plt.show()
pv5 = place.pivot_table(index = 'specialisation', columns = 'status', values = 'gender', aggfunc = 'count')
pv5
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier

place_new = place.drop('salary',axis = 1).reset_index()
place_new = place_new.replace(['Placed','Not Placed'],[1,0])
# print(place_new,'place_new')
place_cat = place_new[['gender','ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation']]
place_num = place_new[['ssc_p','hsc_p','degree_p','etest_p','mba_p']]

X = place_new.drop(['status'],axis = 1).reset_index().drop(['sl_no'],axis = 1)
y = place_new.iloc[:,-1]
from sklearn.compose import ColumnTransformer

num_attribs = list(place_num)
cat_attribs = list(place_cat)
# print(num_attribs,"***num_Attribs***")
# print(cat_attribs,"***cat_attribs***")
num_transformer = Pipeline(steps = [('scaler', StandardScaler())])

cat_transformer = Pipeline(steps = [('onehot',OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers = [('num',num_transformer, num_attribs),
                                                 ('cat',cat_transformer,cat_attribs)])

param_grid = [{'n_estimators':[1,10,100],'max_features': [2, 4, 6, 8]},
              {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}]

forest_clf = RandomForestClassifier()

# grid_search = GridSearchCV(forest_clf, param_grid, cv=5,
# scoring='neg_mean_squared_error',
# return_train_score=True)

clf = Pipeline(steps = [('preprocessor',preprocessor),('grid_search',GridSearchCV(forest_clf, param_grid, cv=5,
scoring='roc_auc',return_train_score=True))])

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)

for train_index, test_index in split.split(X,y):
    #print(train_index,(test_index))
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

# print(X_train.head(),"***Xtrain***",'\n*************************************\n',y_train.head(),'***Y_train***')
# print(X_test.head(),'X_test','\n*******************\n',y_test.head(),'y_test')
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
#print(y_pred)
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))

print("model score: %.3f" % clf.score(X_test, y_test))

