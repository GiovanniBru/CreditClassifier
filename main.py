#!/usr/bin/python
# vim: set fileencoding=unicode-escape :

from credit_data import CreditData
from credit_classifier import CreditClassifier

from sklearn.model_selection import train_test_split

cd = CreditData()
X, Y = cd.get_data("data_train_final1.csv")

cc = CreditClassifier()
#cc.create_model(X, Y)

cc.load_model()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1)
cc.test_model(x_test, y_test)

print(X[0])
print(Y[0])
pred = cc.predict(x_test)
for (cla, prob) in pred:
  print("Classificação: " + str(cla))
  print("Probabilidade: " + str(prob))
