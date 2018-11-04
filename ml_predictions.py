import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns
from confusion_matrix import plot_confusion_matrix as plt_cnf_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Load datasets
red_wine   = pd.read_csv('winequality-red.csv',   sep=';')
white_wine = pd.read_csv('winequality-white.csv', sep=';')
#create class
red_wine['wine_type']='red'
white_wine['wine_type']='white'
# create a new categorical(takes a limited number of possible values)
red_wine['quality_label'] = red_wine['quality'].apply(lambda value: ('low' if value <= 5 else 'medium') if value <= 7 else 'high')
white_wine['quality_label'] = white_wine['quality'].apply(lambda value: ('low' if value <= 5 else 'medium') if value <= 7 else 'high')
# convert to categorical datatype(fixed range of allowed values using pandas
red_wine['quality_label'] = pd.Categorical(red_wine['quality_label'], categories=['low', 'medium', 'high'])
# value is what you feed the lambda function
white_wine['quality_label'] = pd.Categorical(white_wine['quality_label'], categories=['low', 'medium', 'high'])

# convert the classification label to a value for easier processing
#Option 1 USING LABDA

#red_wine['wine_type'] = red_wine['wine_type'].apply(lambda value: (1 if value =='red' else 0))
#Option 2 using get dummies. This will can be used on the combined dataset or red and white wine
# option 3 One Hot Encoding
#wine_type=pd.get_dummies(red_wine['wine_type'],drop_first=False)
# preview the count distribution
print(red_wine['quality_label'].value_counts())
print(red_wine.head())
# combine the datasets for easier processing
wine=pd.concat([red_wine,white_wine],axis=0)
wine['wine_type']=wine['wine_type'].apply(lambda value: (1 if value =='red' else 0))


#perform some pre-processing on the quality and quality label columns
# later will test which performs better
# CONSIDER Low-1 Medium 2 and High-3
wine['quality_label']=wine['quality_label'].apply(lambda value: (1 if value =='low' else 2)  if value != 'high' else 3)
print(wine.tail())

# ML Implementation
#wine.loc[:, wine.columns != 'wine_type'] Feature-set

trainX, testX, trainy, testy = model_selection.train_test_split(wine.loc[:, wine.columns != 'wine_type'], wine['wine_type'], test_size=0.3,random_state=1)
# Train, Test Split
# make a naive prediction. This predicts all of the test data to be one value of the label
# in this case, all the test data will be predicted to be 0 in the 1st round and 1 in round 2
def naive_prediction(testX, value):
	return [value for x in range(len(testX))]

# evaluate skill of predicting each class value
for value in [0, 1]:
	# forecast
	yhat = naive_prediction(testX, value)
	# evaluate
	score = accuracy_score(testy, yhat)
	# summarize
	print('Naive=%d score=%.3f' % (value, score))

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, trainX, trainy, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms by accuracy measures during the 10-fold validation
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
#plt.boxplot(results)
sns.violinplot(data=results,ax=ax)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset
clf = SVC()
clf.fit(trainX, trainy)
predictions = clf.predict(testX)

print('accuracy score',accuracy_score(testy, predictions))
# from sklearn
print('Confusion matrix from sklearn\n')
print(confusion_matrix(testy, predictions))
# custom confusion matrix
plt.figure()
plt_cnf_matrix(confusion_matrix(testy, predictions),classes=['white-wine','red-wine']
			   ,title='Normalized confusion matrix',normalize=True)
plt.show()
print('SKLEARN CLASSIFICATION REPORT\n')
print(classification_report(testy, predictions))