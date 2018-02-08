from sklearn import tree

# Weight | Texture | Label
# 140    | Smooth  | Apple
# 130    | Smooth  | Apple
# 150    | Bumpy   | Orange
# 170    | Bumpy   | Orange

# 0 = Bumpy , 1 = Smooth
features = [
	[140,1],
	[130,1],
	[150,0],
	[170,0]
]

# Apple = 0, Orange = 1
labels = [0,0,1,1]

# Classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print clf.predict([[120,1]])