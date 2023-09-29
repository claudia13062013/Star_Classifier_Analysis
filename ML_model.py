import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold, cross_val_score
import joblib

# get ready with dataset:
df = pd.read_csv("star_classification.csv", sep=",")
df.drop(['obj_ID', 'delta', 'alpha',  'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'spec_obj_ID', 'fiber_ID'], axis=1, inplace=True)
df = df.rename(columns={'class': 'classes'})
df = df[[col for col in df.columns if col != 'classes'] + ['classes']]
print(df.head())

# creating dependent variable class - factorize :
factor = pd.factorize(df['classes'])
df.classes = factor[0]
definitions = factor[1]
print(df.classes.head())
print(definitions)
print(df.head(5))

# cleaning the dataset :
df = df.loc[(df["u"] >= df["u"].quantile(0.025)) & (df["u"] <= df["u"].quantile(0.975)) & (df["g"] >= df["g"].quantile(0.025)) & (df["g"] <= df["g"].quantile(0.975)) & (df["r"] >= df["r"].quantile(0.025)) & (df["r"] <= df["r"].quantile(0.975)) & (df["i"] >= df["i"].quantile(0.025)) & (df["i"] <= df["i"].quantile(0.975)) & (df["z"] >= df["z"].quantile(0.025)) & (df["z"] <= df["z"].quantile(0.975)) & (df["redshift"] >= df["redshift"].quantile(0.025)) & (df["redshift"] <= df["redshift"].quantile(0.975))]

# extracting input and output:
x = df.iloc[:, 0:8].values
y = df.iloc[:, 8].values
print('The independent features set: ')
print(x[:5, :])
print('The dependent variable: ')
print(y[:5])

# trying to make an equal amount of every class:
sns.countplot(x=y, palette='pastel')
plt.title("Class:", fontsize=12)
plt.show()

sm = SMOTE(random_state=42)
x, y = sm.fit_resample(x, y)

sns.countplot(x=y, palette='pastel')
plt.title("Class:", fontsize=12)
plt.show()

# splitting data into train and test :
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=21)

# feature scaling:
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# training model - random forest classification:
classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)

# predicting test results:
y_pred = classifier.predict(x_test)
c_score = classifier.score(x_test, y_test)

# reversing factorization:
reverse_factor = dict(zip(range(3), definitions))
y_test = np.vectorize(reverse_factor.get)(y_test)
y_pred = np.vectorize(reverse_factor.get)(y_pred)

# confusion matrix:
print(pd.crosstab(y_test, y_pred, rownames=['true classes'], colnames=['predicted classes']))

print(list(zip(df.columns[0:8], classifier.feature_importances_)))

# accuracy of the ML model with the Random Forest Classifier:
score = np.mean(c_score)
print('Accuracy : %.3f' % score)
print(classification_report(y_test, y_pred))

# cross validation score:
# k_folds = KFold(n_splits=10)
# scores = cross_val_score(classifier, x, y, cv=k_folds)
# print("Cross Validation Scores: ", scores)
# print("Average CV Score: ", scores.mean())
# print("Number of CV Scores used in Average: ", len(scores))

# Saving the model
joblib.dump(classifier, 'ML_model_obj_classifier.pkl')

# Loading the model from the file
classifier_joblib = joblib.load('ML_model_obj_classifier.pkl')

# Using the loaded model to make predictions
print("After save:", classifier_joblib.predict(x_test))
