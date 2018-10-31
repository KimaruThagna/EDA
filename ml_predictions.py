import pandas as pd
from pandas.plotting import scatter_matrix
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
red_wine['wine_type'] = red_wine['wine_type'].apply(lambda value: (1 if value =='red' else 0))
#Option 2 using get dummies.
wine_type=pd.get_dummies(red_wine['wine_type'],drop_first=False)

# preview the count distribution
print(red_wine['quality_label'].value_counts())
print(red_wine.head())