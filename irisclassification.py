# classification of iris flowers based on sepal and pedal characteristics using a random forest classifier

from sklearn.datasets import load_iris
iris = load_iris()

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10)

#train
model.fit(iris.data, iris.target)
print(iris.target)
#extract tree
estimator = model.estimators_[5]

#visualize
from sklearn.tree import export_graphviz

#export dot file
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = iris.feature_names, 
                class_names = iris.target_names, 
                rounded = True, proportion = False, 
                precision = 2, filled = True)
                
from subprocess import call
call(['dot','-Tpng', 'tree.dot', '-o', 'tree.png','-Gdpi=600'])
