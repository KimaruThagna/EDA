import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler# for scaling data
from wines import * # allowing us to use previous variables

fig, (ax) = plt.subplots(1, 1, figsize=(10,6))
# Compute pairwise correlation of Dataframe's attributes
corr = wine.corr()

hm = sns.heatmap(corr,
                 ax=ax,           # Axes in which to draw the plot, otherwise use the currently-active Axes.
                 cmap="coolwarm", # Color Map.
                 #square=True,    # If True, set the Axes aspect to “equal” so each cell will be square-shaped.
                 annot=True,
                 fmt='.2f',       # String formatting code to use when adding annotations.
                 #annot_kws={"size": 14},
                 linewidths=.05)

fig.subplots_adjust(top=0.93)
fig.suptitle('Wine Attributes Correlation Heatmap',
              fontsize=14,
              fontweight='bold')
plt.show()
# Attributes of interest as of now
cols = ['density',
        'residual sugar',
        'total sulfur dioxide',
        'free sulfur dioxide',
        'fixed acidity']
# another method of observing correlation is using scatter plots
pp = sns.pairplot(wine[cols],
                  size=1.8, aspect=1.2,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kws=dict(shade=True), # "diag" adjusts/tunes the diagonal plots
                  diag_kind="kde") # use "kde" for diagonal plots

fig = pp.fig
fig.subplots_adjust(top=0.93, wspace=0.3)
fig.suptitle('Wine Attributes Pairwise Plots',
              fontsize=14, fontweight='bold')

plt.show()
#fitting linear regression to the scatter plots to give further information
pp = sns.pairplot(wine[cols],
                  diag_kws=dict(shade=True), # "diag" adjusts/tunes the diagonal plots
                  diag_kind="kde",# use "kde" for diagonal plots
                  kind="reg") # linear regression to the scatter plots

fig = pp.fig
fig.subplots_adjust(top=0.93, wspace=0.3)
fig.suptitle('Wine Attributes Pairwise Plots', fontsize=14, fontweight='bold')
plt.show()


#multivariate data can also be represented using parallel coordinates but this would require
#scaling of data as the data columns were measured on different scales

subset_df = wine[cols]

ss = StandardScaler() # scaling of two data sets for them to be represented on shared axis
scaled_df = ss.fit_transform(subset_df)
scaled_df = pd.DataFrame(scaled_df, columns=cols)
final_df = pd.concat([scaled_df, wine['wine_type']], axis=1)
#print(final_df.head())



fig = plt.figure(figsize=(12, 10))
title = fig.suptitle("Parallel Coordinates", fontsize=18)
fig.subplots_adjust(top=0.93, wspace=0)

pc = pd.plotting.parallel_coordinates(final_df,
                          'wine_type',
                          color=('skyblue', 'firebrick'))
plt.show()
# stack bars side by side when doing a simple comparison in count
fig = plt.figure(figsize=(10, 7))
cp = sns.countplot(data=wine,
                   x="quality",
                   hue="wine_type",
                   palette={"red": "#FF9999", "white": "#FFE888"})

# using facetGrid- maps a pandas dataframe to a matplotlib obj

fig = plt.figure(figsize=(10,8))
title = fig.suptitle("Sulphates Content in Wine", fontsize=14)
fig.subplots_adjust(top=0.93, wspace=0.3)

ax = fig.add_subplot(1,1,1)
ax.set_xlabel("Sulphates")
ax.set_ylabel("Frequency")

g = sns.FacetGrid(data=wine,
                  hue='wine_type',
                  palette={"red": "r", "white": "y"})

g.map(sns.distplot, 'sulphates',
      kde=True, bins=15, ax=ax)

ax.legend(title='Wine Type')
plt.show()
plt.close(2)

# violin plots(can substitute boxplots)

f, (ax) = plt.subplots(1, 1, figsize=(12, 4))
f.subtitle('Wine Quality - Sulphates Content', fontsize=14)
sns.violinplot(data=wine,
               x="quality",
               y="sulphates",
               ax=ax)
# also box plots can be used here
# sns.boxplot(data=wine,
#             x="quality",
#             y="sulphates",
#             ax=ax)
ax.set_xlabel("Wine Quality",size=12,alpha=0.8)
ax.set_ylabel("Wine Sulphates",size=12,alpha=0.8)
plt.show()