import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# read data set:
data = pd.read_csv("star_classification.csv", sep=",")
pd.set_option('display.max_columns', None)

# looking at the distribution of every object to see how different are :
filter1 = data['class'] == 'STAR'
filter2 = data['class'] == 'GALAXY'
filter3 = data['class'] == 'QSO'
fig, ax = plt.subplots(3, 1, figsize=(12, 14))
ax[0].scatter(data['alpha'].where(filter1), data['delta'].where(filter1), marker='.', color='green')
ax[0].set_title('stars')
ax[1].scatter(data['alpha'].where(filter2), data['delta'].where(filter2), marker='.', color='green')
ax[1].set_title('galaxies')
ax[2].scatter(data['alpha'].where(filter3), data['delta'].where(filter3), marker='.', color='green')
ax[2].set_title('qso')
plt.show()

# into correlation with every class:

data["class"] = np.where(data["class"] == 'STAR', 1, 0)
# pearson and spearman correlation :
corr_star_pears = data.corr(numeric_only=False)
corr_star_spear = data.corr('spearman', numeric_only=False)

plt.subplots(figsize=(12, 9))
sns.heatmap(corr_star_pears, vmax=0.8, fmt='.1f', annot=True)

plt.show()

plt.subplots(figsize=(12, 9))
sns.heatmap(corr_star_spear, vmax=0.8, fmt='.1f', annot=True)

plt.show()

# I will use Spearman correlation because of outliers and non always linear data
data = pd.read_csv("star_classification.csv", sep=",")
data["class"] = np.where(data["class"] == 'QSO', 1, 0)
corr_qso = data.corr('spearman', numeric_only=False)

plt.subplots(figsize=(12, 9))
sns.heatmap(corr_qso, vmax=0.8, fmt='.1f', annot=True)

plt.show()

data = pd.read_csv("star_classification.csv", sep=",")
data["class"] = np.where(data["class"] == 'GALAXY', 1, 0)
corr_galaxy = data.corr('spearman', numeric_only=False)

plt.subplots(figsize=(12, 9))
sns.heatmap(corr_galaxy, vmax=0.8, fmt='.1f', annot=True)

plt.show()
data = pd.read_csv("star_classification.csv", sep=",")

# here I am using biserial correlation for a binary - continuous  data:

# qso correlations analysis:
# 1) with redshifts

data["class"] = np.where(data["class"] == 'QSO', 1, 0)
corr_gso_redsh2 = stats.pointbiserialr(data['class'], data['redshift'])
print("corr biserial:", corr_gso_redsh2)
# correlation is statistically significant

sns.boxplot(data=data, x='class', y='redshift')
plt.title('0 is galaxies and stars, 1 is qso ')
plt.show()

# 2) with 'i'
corr_qso_i = stats.pointbiserialr(data['class'], data['i'])
print('gso with i corr:', corr_qso_i)
# correlation is statistically significant

# 3) with 'r'
corr_qso_r = stats.pointbiserialr(data['class'], data['r'])
print('gso with r corr:', corr_qso_r)
# correlation is statistically significant

# Galaxies correlations analysis:
# galaxy and redshift negative corr.:
data = pd.read_csv("star_classification.csv", sep=",")
data["class"] = np.where(data["class"] == 'GALAXY', 1, 0)
corr_gal_redsh = stats.pointbiserialr(data['class'], data['redshift'])
print('galaxy with redshift neg. corr.:', corr_gal_redsh)
# correlation is statistically significant

# galaxy and r corr:
data_corr_galaxy = data.loc[(data["r"] >= data["r"].quantile(0.025)) & (data["r"] <= data["r"].quantile(0.975))]
corr_gal_r = stats.pointbiserialr(data_corr_galaxy['class'], data_corr_galaxy['r'])
print('galaxy with r neg. corr.:', corr_gal_r)

sns.boxplot(data=data_corr_galaxy, x='class', y='r')
plt.show()

# Stars correlations analysis:
# stars and redshift negative corr.:
data = pd.read_csv("star_classification.csv", sep=",")
data["class"] = np.where(data["class"] == 'STAR', 1, 0)
corr_star_redsh = stats.pointbiserialr(data['class'], data['redshift'])
print('stars and redshift neg. corr.:', corr_star_redsh)
# correlation is statistically significant

# stars and r corr:
corr_star_r = stats.pointbiserialr(data['class'], data['r'])
print('stars and r neg. corr.:', corr_star_r)

# seeing correlation results and analysis we can clearly drop from dataset some features
