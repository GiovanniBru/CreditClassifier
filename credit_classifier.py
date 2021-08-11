import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#from tqdm import tqdm # barra de progresso
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pickle

class CreditClassifier:
  def __init__(self):
    self.model = None
    self.vmax_ = []
    self.vmin_ = []

  def create_model(self, X, Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1)
    self.compute_max_min(x_train)
    self.normalize(x_train)
    self.cross_validation(x_train, y_train)

    self.test_model(x_test, y_test)
    self.persist_model()

  def cross_validation(self, x_train, y_train):
    param_grid = {'C': [1, 5, 10, 20, 30, 40],
            'gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]}
    grid = GridSearchCV(estimator=SVC(kernel='rbf', probability=True, class_weight={0: 3.8, 1: 1}), param_grid=param_grid, cv=5)

    grid.fit(x_train, y_train)
    print(grid.best_params_)

    y_in = grid.predict(x_train)
    print(classification_report(y_train, y_in))

    self.model = grid.best_estimator_


  def load_model(self):
    self.model = pickle.load(open('model/model.sav', 'rb'))
    self.vmax_ = pickle.load(open('model/max.sav', 'rb'))
    self.vmin_ = pickle.load(open('model/min.sav', 'rb'))


  def persist_model(self):
    pickle.dump(self.model, open('model/model.sav', 'wb'))
    pickle.dump(self.vmax_, open('model/max.sav', 'wb'))
    pickle.dump(self.vmin_, open('model/min.sav', 'wb'))

  def train_model(self, x_train, y_train, C_, gamma_):
    self.model = SVC(kernel='rbf', C=C_, gamma=gamma_)
    self.model.fit(x_train, y_train)
    y_pred = self.model.predict(x_test)
    acc_train = accuracy_score(y_train, y_pred)

  def predict(self, x_test):
    if(self.model == None):
      return None
    self.normalize(x_test)
    prob = self.model.predict_proba(x_test)
    pred = self.model.predict(x_test)
    return [(e, o[int(e)]) for e, o in zip(pred, prob)]

  def test_model(self, x_test, y_test):
    if(self.model == None):
      return None
    self.normalize(x_test)
    y_pred = self.model.predict(x_test)
    print(classification_report(y_test, y_pred))


  def get_outliers(self, X_, Y):
    X = np.array(X_)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    Y_clustered = kmeans.fit_predict(X)

    print(confusion_matrix(Y, Y_clustered))
    i = 0
    not_outliers = []
    for i in range(len(X)):
        if Y_clustered[i] == Y[i]:
            not_outliers.append(i)

    return not_outliers

  def split_data(self, x, y):
    x0 = []
    x1 = []
    for i in range(len(y)):
        if y[i] == 1:
            x1.append(x[i])
        else:
            x0.append(x[i])

    prop = len(x0) / len(x1)

    size0 = int(len(x0) * ((prop - 1) / prop))
    size1 = int(len(x1) * ((prop - 1) / prop))

    x0 = shuffle(x0, random_state = 1)
    x_train0 = x0[:size0]
    x_test0 = x0[size0:]
    y_train0 = [0] * len(x_train0)
    y_test0 = [0] * len(x_test0)

    x1 = shuffle(x1, random_state=1)
    x_train1 = x1[:size1]
    x_test1 = x1[size1:]
    y_train1 = [1] * len(x_train1)
    y_test1 = [1] * len(x_test1)

    x_train = np.concatenate((x_train0, x_train1))
    x_test = np.concatenate((x_test0, x_test1))
    y_train = np.concatenate((y_train0, y_train1))
    y_test = np.concatenate((y_test0, y_test1))


    return x_train, y_train, x_test, y_test

  def normalize(self, X):
    col = [0,1,2,4,19,20,24,25,26]

    for c in col:
        for i in range(len(X)):
            X[i][c] = (X[i][c] - self.vmin_[c]) / (self.vmax_[c] - self.vmin_[c])
            #X[i][c] = scaled[i]

  def compute_max_min(self, x_train):
    lin = len(x_train)
    col = len(x_train[0])

    self.vmax_ = []
    self.vmin_ = []
    for c in range(col):
      max = float('-inf')
      min = float('inf')

      for l in range(lin):
        if(max < x_train[l][c]):
          max = x_train[l][c]
        if(min > x_train[l][c]):
          min = x_train[l][c]

      self.vmax_.append(max)
      self.vmin_.append(min)
