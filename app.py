import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Load the iris dataset
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.float64)

# Create a pipeline with a scaler and a linear SVM
svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge")),
])

# Train the model
svm_clf.fit(X, y)

new_data = []
with open('svm.txt', 'r') as file:
    for line in file:
        values = line.strip().split(',')
        new_data.append([float(values[0]), float(values[1])])

new_data = np.array(new_data)

# Use the model to predict the class of the new data points
predictions = svm_clf.predict(new_data)


print(predictions)