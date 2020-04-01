from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import sklearn.multiclass
import xgboost as xgb

n_jobs=5

Split = namedtuple('Split', ['X', 'y'])

def split(data: pd.DataFrame):
	return Split(data.drop(columns='target').copy(), data['target'].copy())


def plot_confusion_matrix(classifier):
	kwargs = dict(annot=True, square=True, fmt='d')
	ax = sns.heatmap(classifier.confusion_matrix, **kwargs)
	ax.set(xlabel='predicted', ylabel='true')


def roc_auc_score(y_true: pd.Series, y_score:pd.Series) -> float: 
	return sklearn.metrics.roc_auc_score(
		y_true.values, y_score, multi_class='ovr', average='macro'
	)


def plot_roc_curve(classifier):
	for class_name in classifier.classes:
		fpr, tpr, _ = sklearn.metrics.roc_curve(
			classifier.y_true==class_name, classifier.y_score[class_name]
			)
		plt.plot(fpr, tpr, label=class_name)
	plt.plot([0,1], [0,1], linewidth=2, color='black')
	plt.xlabel('fpr')
	plt.ylabel('tpr')
	plt.legend()
	plt.xlim(0, 1)
	plt.ylim(0, 1)


class XGBClassifier:
	def __init__(self, n_jobs=n_jobs, random_state=0):
		self.clf = xgb.XGBClassifier(n_jobs=n_jobs, random_state=random_state)
		self.label_encoder = sklearn.preprocessing.LabelEncoder()

	def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
		y = self.label_encoder.fit_transform(y)
		self.classes = self.label_encoder.classes_
		self.clf.fit(X.to_numpy(), y) # Only np.arrays work without problems.
		
	def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
		return pd.DataFrame(
			data=self.clf.predict_proba(X.values), 
			index=X.index,
			columns=self.label_encoder.classes_
			)

	def predict(self, X: pd.DataFrame) -> pd.Series:
		data = self.label_encoder.inverse_transform(self.clf.predict(X.values))
		return pd.Series(data, index=X.index)

	def test(self, X: pd.DataFrame, y: pd.Series) -> None:
		self.y_true = y
		self.y_predict = self.predict(X)
		self.y_score = self.predict_proba(X)
		self.confusion_matrix = pd.DataFrame(
			sklearn.metrics.confusion_matrix(self.y_true, self.y_predict), 
			self.label_encoder.classes_, self.label_encoder.classes_
		)
		self.roc_auc_score = roc_auc_score(self.y_true, self.y_score)