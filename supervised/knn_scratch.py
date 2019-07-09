import numpy as np
import operator
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

"""
X_test point finds K nearest neighbour
select the most voted class

"""
def euclidean_distance(x,y):
    """
    x = vector 1
    y = vector 2
    """
    return np.sqrt(np.sum((x-y)**2))

def get_neighbours(X_train, X_test_instance, k):
    distances = []
    neighbours = []
    X_train_row = X_train.shape[0]
    for i in range(X_train_row):
        distance = euclidean_distance(X_train[i], X_test_instance)
        distances.append((i,distance))
    distances.sort(key=operator.itemgetter(1))
    for d in range(k):
        neighbours.append(distances[d][0])
    return neighbours

def voting_fn(output, y_train):
    classVotes = {}
    for i in range(len(output)):
        if y_train[output[i]] in classVotes:
            classVotes[y_train[output[i]]] +=1
        else:
            classVotes[y_train[output[i]]] = 1
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def predict(X_train, X_test, Y_train, Y_test, k):
    output_class = []
    for i in range(0, X_test.shape[0]):
        output = get_neighbours(X_train, X_test[i], k)
        predictedClass = voting_fn(output, Y_train)
        output_class.append(predictedClass)
    return output_class


if __name__ == "__main__":
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)
    y_predict = predict(X_train, X_test, y_train, y_test, 3)
    print(confusion_matrix(y_test, y_predict))
