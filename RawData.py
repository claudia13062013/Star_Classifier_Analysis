import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import Angle

# read data set:
data = pd.read_csv("star_classification.csv", sep=",")
pd.set_option('display.max_columns', None)

# basic look on data:
print(data.head(10))
print(data.info())
print(data.describe().transform)
print(data['class'].unique())
# notes to remember: look into 'redshift' plot, filters('u', 'g'...) plot

# small plot to see object's location on the sky and possibility of patterns :
plt.scatter(data['alpha'], data['delta'], marker='.', color='green')
plt.show()

# making more clear visual representation:

a = Angle(data['alpha'] * u.deg)
b = Angle(data['delta'] * u.deg)
fig, axes = plt.subplots(2, 1, figsize=(8, 10), subplot_kw={'projection': 'aitoff'})
axes[0].set_title("Objects")
axes[0].plot(a.wrap_at(180*u.deg).radian, b.radian, linestyle='none', marker='.')
axes[1].set_title(" ")
axes[1].plot(0, 0, linestyle='none', marker='.')

plt.show()

# summing every class:
galaxy_class = sum(data['class'] == 'GALAXY')
print(galaxy_class)
star_class = sum(data['class'] == 'STAR')
print(star_class)
qso_class = sum(data['class'] == 'QSO')
print(qso_class)
classes = [galaxy_class, star_class, qso_class]
names = ['GALAXY', 'STAR', 'QSO']

# making plot:
bwidth = 1
bars = np.arange(len(classes))
plt.bar(bars, classes, color=['purple', 'blue', 'green'], width=bwidth, label='sum of each class')
plt.xticks([r for r in range(len(classes))], names)
plt.show()
# most objects are galaxies
# more analysis into features of data in file: AnalysisEDAData.py
