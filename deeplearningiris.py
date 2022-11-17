# iris classification using deep learning model

from sklearn import svm
from sklearn.datasets import load_iris
iris = load_iris()

x= iris['data'][0:126]
y= iris['target'][0:126]

x_test = iris['data'][125:]
y_test = iris['target'][125:]

model = svm.SVC(kernel='linear')

model.fit(x, y)
model.score(x, y)
predicted=model.predict(x_test)

testScore = 0
for ind, predict in enumerate(predicted):
    if predict == y_test[ind]:
        testScore +=1

print("Accuracy:", str(testScore/len(y_test)))
