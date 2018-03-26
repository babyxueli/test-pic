# encoding=utf-8
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  #  用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    #  用来正常显示正负号
import seaborn as sns
sns.set_style('whitegrid')


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

# titanic_df = pd.read_csv("/home/lxl/Documents/titanic-master/data/train.csv")
# test_df    = pd.read_csv("/home/lxl/Documents/titanic-master/data/test.csv")
# # preview the data
# # print titanic_df.head()
# # print test_df.head()
train = pd.read_csv('/home/lxl/Documents/titanic-master/data/train.csv')
test = pd.read_csv('/home/lxl/Documents/titanic-master/data/test.csv')
# print train.describe()
# print test.describe()

# # drop unnecessary columns, these columns won't be useful in analysis and prediction
# titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'], axis=1)
# test_df    = test_df.drop(['Name','Ticket'], axis=1)
train = train.drop(['PassengerId','Name','Ticket'],axis=1)
test = test.drop(['Name','Ticket'],axis=1)



#  01 Embarked
train['Embarked'] = train['Embarked'].fillna('S')
# sns.factorplot('Survived','Embarked',data=train,size=4,aspect=3) 注意x、y的顺序，见下面的代码
# sns.factorplot('Embarked','Survived',data=train,size=4,aspect=3)
# sns.factorplot('Embarked',data=train,kind='count',order=['S','C','Q'],ax=axis)
# sns.factorplot('Survived',hue='Embarked',data=train,kind='count',order=[1,0],ax=axis)
# fig,(axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))
# sns.countplot(x='Embarked',data=train,order=['S','C','Q'],ax=axis1)
# sns.countplot(x='Survived',hue='Embarked',data=train,order=[1,0],ax=axis2)
# embark_perc = train[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean()
# # sns.barplot(x='Embarked',y='Survived',data=embark_perc,order=['S','C','Q'],ax=axis3) 等同于下面的代码
# sns.barplot('Embarked','Survived',data=embark_perc,order=['S','C','Q'],ax=axis3)
# plt.show()


# # Either to consider Embarked column in predictions,and remove "S" dummy variable,

# 标准化 用0\1 来代替
embark_dummies_train = pd.get_dummies(train['Embarked'])
embark_dummies_train.drop(['S'],axis=1,inplace=True)
# print embark_dummies_train
embark_dummies_test = pd.get_dummies(test['Embarked'])
embark_dummies_test.drop(['S'],axis=1,inplace=True)


train = train.join(embark_dummies_train)
train.drop(['Embarked'],axis=1,inplace=True)
test = test.join(embark_dummies_test)
test.drop(['Embarked'],axis=1,inplace=True)


#
##  02Fare
# # only for test_df, since there is a missing "Fare" values

# train['Fare'].fillna(train['fare'].median,inplace=True)
test['Fare'].fillna(test['Fare'].median(),inplace=True)
train['Fare'] = train['Fare'].astype(int)
test['Fare'] = test['Fare'].astype(int)
# print test.head()
# # get fare for survived & didn't survive passengers

fare_not_survived = train['Fare'][train['Survived']==0]
fare_survived = train['Fare'][train['Survived']==1]


# # get average and std for fare of survived/not survived passengers

average_fare = DataFrame([fare_not_survived.mean(),fare_survived.mean()])
std_fare = DataFrame([fare_not_survived.std(),fare_survived.std()])
# train['Fare'].plot(kind='hist',figsize=(15,3),bins=100,xlim=(0,50))
# plt.show()
average_fare.index.names = std_fare.index.names = ['Survived']
# average_fare.plot(yerr=std_fare,kind='bar',legend=False)
# plt.show()



# # Age
# fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))
# axis1.set_title('Original Age values - Titanic')
# axis2.set_title('New Age values - Titanic')
# # axis3.set_title('Original Age values - Test')
# # axis4.set_title('New Age values - Test')

#Age
# fig,(axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
# axis1.set_title('Original Age values - Train')
# axis2.set_title('New Age values - test')

average_age_titanic = train['Age'].mean()
std_age_titanic = train['Age'].std()
count_nan_age_titanic = train['Age'].isnull().sum()
average_age_test = test['Age'].mean()
std_age_test = test['Age'].std()
count_nan_age_test = test['Age'].isnull().sum()

rand_1 = np.random.randint(average_age_titanic-std_age_titanic,average_age_titanic+std_age_titanic,size=count_nan_age_titanic)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size=count_nan_age_test)

# train['Age'].dropna().astype(int).hist(bins=70,ax=axis1)
# test['Age'].dropna().astype(int).hist(bins=70,ax=axis1)
train['Age'][np.isnan(train['Age'])] = rand_1
test['Age'][np.isnan(test['Age'])] = rand_2
train['Age'] = train['Age'].astype(int)
test['Age'] = test['Age'].astype(int)
# train['Age'].hist(bins=70,ax=axis2)
# test['Age'].hist(bins=70,ax=axis2)
# plt.show()



# # continue with plot Age column, peaks for survived/not survived passengers by their age
# facet = sns.FacetGrid(train,hue='Survived',aspect=4)
# facet.map(sns.kdeplot,'Age',shade=True)
# facet.set(xlim=(0,train['Age'].max()))
# facet.add_legend()
# fig,axis1 = plt.subplots(1,1,figsize=(18,4))

average_age = train[['Age','Survived']].groupby(['Age'],as_index=False).mean()
# sns.barplot(x='Age',y='Survived',data=average_age)
# plt.show()


# # Cabin
# # It has a lot of NaN values, so it won't cause a remarkable impact on prediction
train.drop('Cabin',axis=1,inplace=True)
test.drop('Cabin',axis=1,inplace=True)

# # Family,       Instead of having two columns Parch & SibSp
train['Family'] = train['Parch'] + train['SibSp']
# print train['Family']
train['Family'].loc[train['Family'] > 0] =1
train['Family'].loc[train['Family'] == 0] = 0
# print train['Family']
test['Family'] = test['Parch'] + test['SibSp']
test['Family'].loc[train['Family'] > 0] = 1
test['Family'].loc[train['Family'] ==0] = 0

train = train.drop(['SibSp','Parch'],axis=1)
test = test.drop(['SibSp','Parch'],axis=1)

# fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))
# sns.factorplot('Family',data=titanic_df,kind='count',ax=axis1)
# sns.countplot(x='Family', data=titanic_df, order=[1,0], ax=axis1)
# fig,(axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))
# sns.factorplot('Family',data=train,kind='count',ax=axis1)
# sns.countplot(x='Family',data=train, order=[1,0], ax=axis1)
family_perc = train[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()
# sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)
# axis1.set_xticklabels(["With Family","Alone"], rotation=0)
# plt.show()


# # Sex
# # As we see, children(age < ~16) on aboard seem to have a high chances for Survival.
# # So, we can classify passengers as males, females, and child
def get_person(passenger):
    age, sex = passenger
    return 'child' if age < 16 else sex

train['Person'] = train[['Age', 'Sex']].apply(get_person, axis=1)
test['Person'] = test[['Age', 'Sex']].apply(get_person, axis=1)
train.drop(['Sex'], axis=1, inplace=True)
test.drop(['Sex'], axis=1, inplace=True)
# # create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
person_dummies_titanic = pd.get_dummies(train['Person'])
person_dummies_titanic.columns = ['Child', 'Female', 'Male']
person_dummies_titanic.drop(['Male'], axis=1, inplace=True)
person_dummies_test = pd.get_dummies(test['Person'])
person_dummies_test.columns = ['Child', 'Female', 'Male']
person_dummies_test.drop(['Male'], axis=1, inplace=True)
train = train.join(person_dummies_titanic)
test = test.join(person_dummies_test)

# fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(10, 5))
# sns.factorplot('Person',data=train,kind='count',ax=axis1)
# sns.countplot(x='Person', data=train, ax=axis1)

person_perc = train[["Person", "Survived"]].groupby(['Person'], as_index=False).mean()
# sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male', 'female', 'child'])
train.drop(['Person'], axis=1, inplace=True)
test.drop(['Person'], axis=1, inplace=True)
# plt.show()



# # Pclass
# sns.factorplot('Pclass','Survived',order=[1,2,3], data=train,size=5)
# plt.show()
pclass_dummies_titanic  = pd.get_dummies(train['Pclass'])
pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)

pclass_dummies_test  = pd.get_dummies(test['Pclass'])
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)

train.drop(['Pclass'],axis=1,inplace=True)
test.drop(['Pclass'],axis=1,inplace=True)
train = train.join(pclass_dummies_titanic)
test  = test.join(pclass_dummies_test)

# print train.head()
# print train.describe()

# define training and testing sets
X_train = train.drop("Survived",axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId",axis=1).copy()
Y_test  = test["Survived"]
# X_train, X_test, Y_train, Y_test = train_test_split( X_train,Y_train,X_test , Y_test, train_size=0.7, random_state=0)
# 模型预测
# # Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
print logreg.score(X_test, Y_pred)


# # Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
print  svc.score(X_test, Y_pred)


# # Random Forests
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
print random_forest.score(X_test, Y_pred)

# # Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
print gaussian.score(X_test, Y_pred)

# #using Logistic Regression to get Correlation Coefficient for each feature
coeff_df = DataFrame(train.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])
print coeff_df

