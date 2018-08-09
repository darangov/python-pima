# Save Model Using Pickle
# The example below demonstrates how you can train a logistic regression model on the 
# Pima Indians onset of diabetes dataset, save the model to file and load it to make 
# predictions on the unseen test set (update: download from here).
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle

"""
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
"""

#Data desde archivo local
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(filename, names=names)
#dataset = pandas.read_csv(filename, names=names)

array = dataframe.values
X = array[:,0:8] # Todas las filas, 8 columnas, de la columna 0 a la 7 
Y = array[:,8] # Todas las filas, columna en posicion 8
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on 33%
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# some time later...

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)