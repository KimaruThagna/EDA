WINE ATTRIBUTES CORRELATION HEATMAP
Alcohol content and density are strongly negatively correlated(-0.69)
Free sulphur dioxide and total suplhur dioxide are strongly positively correlated(0.72)
In the pairwise scatter plots with linear regression fitted, the line has a positive slope
hence showing most pairs are +vely correlated.
In the Parallel coordinate Figure,
Red wine seems to have a lower total sulphur dioxide and higher acidity content
compared to white wine which also has a lower density

VIOLIN/BOX PLOTS
They show the probability density of grouped data at different values and can also be
used to identify potential outliers.

#ML PREDICTIONS SCRIPT
Quality label had to be encoded into figures for the algorithms to work
This label is insignificant since the quality column already exists and hence, not necessary to training

As per my runnings, the best algorithm was LDA
LR: 0.981526 (0.004636)
LDA: 0.995821 (0.004765)
KNN: 0.940178 (0.011804)
CART: 0.984386 (0.005245)
NB: 0.976030 (0.008592)
SVM: 0.944137 (0.013777)
