import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from mpl_toolkits.mplot3d import Axes3D

#Load datasets
red_wine   = pd.read_csv('winequality-red.csv',   sep=';')
white_wine = pd.read_csv('winequality-white.csv', sep=';')
# view the dataset's first 5 items
#print(red_wine.head())

#add new attribute(could also work as label
red_wine['wine_type']='red'
white_wine['wine_type']='white'
# to view the unique values of a column
#sorted(red_wine['quality'].unique())
'''
Decide on the quality label values
low<=5   high >7 and medium between 5 and 7
Such a scale would be determined by a professional
'''
# create a new categorical(takes a limited number of possible values)
red_wine['quality_label'] = red_wine['quality'].apply(lambda value: ('low' if value <= 5 else 'medium') if value <= 7 else 'high')
# convert to categorical datatype(fixed range of allowed values using pandas
red_wine['quality_label'] = pd.Categorical(red_wine['quality_label'], categories=['low', 'medium', 'high'])
white_wine['quality_label'] = white_wine['quality'].apply(lambda value: ('low' if value <= 5 else 'medium') if value <= 7 else 'high')
# value is what you feed the lambda function
white_wine['quality_label'] = pd.Categorical(white_wine['quality_label'], categories=['low', 'medium', 'high'])

# preview the count distribution
#print(red_wine['quality_label'].value_counts())
# merging datasets. axis=0 means we are concatinating rows and hence, datasets
# should have same number of columns axis=1 is the other way round
wine=pd.concat([red_wine,white_wine],axis=0)
# reset index ensures that the resulting dataset starts from index 0 drop=True
# restricts the function from adding the previous indices as a column to the new subset
wine=wine.sample(frac=1.0,random_state=42).reset_index(drop=True)
# get a general summary of the data frame
#print(round(wine[wine.columns.values].describe(),2))
subset_attributes = ['residual sugar',
                     'total sulfur dioxide',
                     'sulphates',
                     'alcohol',
                     'volatile acidity',
                     'quality']
redWineSummary=round(red_wine[subset_attributes].describe(),2)
WhiteWineSummary=round(white_wine[subset_attributes].describe(),2)
summary=pd.concat([redWineSummary, WhiteWineSummary], axis=1, #since we're joining columns
          keys=['X Red Wine Statistics',
                'W️ White Wine Statistics'])
#print(summary)

subset2_attributes = ['alcohol', 'volatile acidity', 'pH', 'quality']

lq = round(wine[wine['quality_label'] == 'low'][subset2_attributes].describe(), 2)
# this time, we describe a subset of the data based on a condition imposed on the column quality_label
mq = round(wine[wine['quality_label'] == 'medium'][subset2_attributes].describe(), 2)
hq = round(wine[wine['quality_label'] == 'high'][subset2_attributes].describe(), 2)

summary2=pd.concat([lq, mq, hq], axis=1,
          keys=['L Low Quality Wine',
                'M Medium Quality Wine',
                'H High Quality Wine'])
#print(summary2.head())

# visualization
fig1 = wine.hist(bins=15,
                 color='steelblue',
                 edgecolor='black', linewidth=1.0,
                 xlabelsize=10, ylabelsize=10,
                 xrot=45, yrot=0,
                 figsize=(10,9),
                 grid=False)

plt.tight_layout(rect=(0, 0, 1.5, 1.5))
#plt.show()

# side by side histogram and density plot

fig = plt.figure( figsize=(12,4) )
title = fig.suptitle("Sulphates Content in Wine", fontsize=16, fontweight='bold')
fig.subplots_adjust(top=0.88, wspace=0.3)


# Histogram

ax1 = fig.add_subplot(1,2,1)
ax1.set_xlabel("Sulphates")
ax1.set_ylabel("Frequency")

ax1.text(x=1.2, y=800,
         s=r'$\mu$='+str(round(wine['sulphates'].mean(),2)),
         fontsize=12)

freq, bins, patches = ax1.hist(wine['sulphates'],
                               bins=40,
                               color='darksalmon',
                               edgecolor='darkred', linewidth=1)


# Density Plot
ax2 = fig.add_subplot(1,2,2)
#share x axis with ax1
#ax2 = ax1.twinx() # https://youtu.be/OebyvmZo3w0?t=1m42s
ax2.set_xlabel("Sulphates")
ax2.set_ylabel("Density")
sb.kdeplot(wine['sulphates'], ax=ax2, shade=True, color='forestgreen')

#save
#fig.savefig('suplhates_content_in_wine_side-by-side.jpg')
w_quality = wine['quality'].value_counts() #get values and their count
w_quality = (list(w_quality.index), list(w_quality.values))


fig = plt.figure(figsize=(6, 4))
title2 = fig.suptitle("Wine Quality Frequency", fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.9, wspace=0.3)

ax = fig.add_subplot(1,1,1)
ax.set_xlabel("Quality")
ax.set_ylabel("Frequency")
ax.tick_params(axis='both', which='major', labelsize=8.5)

bar = ax.bar(w_quality[0],
             w_quality[1],
             width=0.85,
             color='plum',
             edgecolor='black', linewidth=1)
plt.show()
