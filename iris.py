import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

# Indexs para remover un tipo de cada flor
test_idx = [0,50,100]

# Nuevo training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# Nuevo testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# Entrenamos el classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# Visualiza el árbol
"""
from sklearn.externals.six import StringIO
import pydot

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,feature_names=iris.feature_names,
					class_names=iris.target_names, filled=True, rounded=True,
					impurity=False)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("iris.pdf")
"""

# Recordar que tenemos 3 flores
flor = 1
print("Flor Actual")
print("Datos de flor ->", test_data[flor])
print()
print("Tipo de flor ->", test_target[flor])

print(" --- * ---- ")

print(iris.feature_names)
print()
print(iris.target_names)