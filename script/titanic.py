import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.kernel_approximation import RBFSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
# data visualization
import seaborn as sns
#matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style


train_data = pd.read_csv("../db/train.csv")
test_data = pd.read_csv("../db/test.csv")
pd.options.mode.chained_assignment = None
'''
# visualisation of the null in
total = train_data.isnull().sum().sort_values(ascending=False)
pourcent1 = train_data.isnull().sum()/train_data.isnull().count()*100
pourcent2 = (round(pourcent1,1).sort_values(ascending=False))
print(pourcent2)
print(total)
'''
##print(train_data.info())


#Embarked, Pclass and Sex
'''
FacetGrid = sns.FacetGrid(train_data, row='Embarked', size=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()
plt.show()
'''
'''
#Pclass: 

sns.barplot(x='Pclass', y='Survived', data=train_data)
plt.show()
'''

'''
#Pclass:
grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()
'''


#Ajout d'une case relatives = Parch + SibSp & not_not =1 pour dire qui n'est pas seul, 0 sinon
data= [train_data,test_data]

for dataset in data: 
    dataset['Relatives']=dataset['Parch'] + dataset['SibSp']
    dataset.loc[dataset['Relatives']>0,'not_alone']=1
    dataset.loc[dataset['Relatives']==0,'not_alone']=0
    dataset['not_alone']=dataset['not_alone'].astype(int)
'''
#Relatives: Combainaison between Parch and SibSp
sns.barplot(x='Relatives', y='Survived', data=train_data)
plt.show()
'''
# Adding the avearage age to the dataset where there is a nan
mean=train_data.Age.mean()
std= train_data.Age.std()
train_data.loc[np.isnan(train_data['Age']),'Age']=np.random.randint(mean-std,mean+std)


# The most commun port is S, so we complete the dataset with S where there is a nan
for dataset in data:
    dataset['Embarked']= dataset['Embarked'].fillna('S')

for dataset in data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr','Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
sex= {'male':1, 'female':2}
embarked ={'S':1, 'Q':2, 'C':3}
for dataset in data:
    dataset.loc[(dataset['Age']<=17.0),"Age"]=0
    dataset.loc[(dataset['Age']<= 24.0) & (train_data['Age']>17.0),'Age']=1
    dataset.loc[(dataset['Age']<= 35.0) & (train_data['Age']>24.0),'Age']=2
    dataset.loc[(dataset['Age']> 35.0),'Age']=3
    dataset['Age']=dataset['Age'].fillna(0)
    dataset['Age']=dataset['Age'].astype(int)
    dataset['Fare']=dataset['Fare'].fillna(0)
    dataset['Fare']=dataset['Fare'].astype(int)
    dataset['Sex']= dataset['Sex'].map(sex)
    dataset['Embarked'] =dataset['Embarked'].map(embarked)
    dataset['Title']=dataset['Title'].map(titles).fillna(5)
    dataset['Title']=dataset['Title'].astype(int)

# the classification of the Fare
for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)

#Fare per person:
for dataset in data: 
    dataset['fare_per_person']=dataset['Fare']/(dataset['Relatives']+1)
    dataset['fare_per_person']=dataset['fare_per_person'].astype(int)

#Combinaison of age and sex :
for dataset in data: 
    dataset['Age_Sex']=dataset['Age']*dataset['Sex']
    dataset['Age_Sex']=dataset['Age_Sex'].astype(int)

train_data=train_data.drop('Ticket',axis=1)
train_data=train_data.drop('Cabin',axis=1)
train_data=train_data.drop('Name',axis=1)
test_data=test_data.drop('Ticket',axis=1)
test_data=test_data.drop('Cabin',axis=1)
test_data=test_data.drop('Name',axis=1)


print(train_data.info())








'''for i in range(0,len(train_data)):
    if(math.isnan(train_data.Age[i])):
        train_data.Age[i]=29

for i in range(0,len(test_data)):
    if(math.isnan(test_data.Age[i])):
        test_data.Age[i]=29
'''


'''
#Sex and age plot : 

survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = train_data[train_data['Sex']=='female']
men = train_data[train_data['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde =False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde =False)
ax.legend()
plt.show()

FacetGrid = sns.FacetGrid(train_data, row='Survived', size=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot,'Age', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()
plt.show()
'''




features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Relatives','not_alone','Title','fare_per_person','Age_Sex']
X=pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
Y = train_data['Survived']


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X, Y)

pred = random_forest.predict(X_test)
rf = RandomForestClassifier(n_estimators=100)

scores = cross_val_score(rf, X, Y, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
output = pd.DataFrame({'PassengerId':test_data.PassengerId, 'Survived':pred})
output.to_csv('../my_submission.csv', index=False)
