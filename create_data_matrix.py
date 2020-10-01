import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics


class MixedModel:
    """
    Model that classifies non-scary tracks, and performs regression to estimate the amount of fear for
    fearful tracks
    """
    def __init__(self, alpha):
        self.classification_clf = linear_model.LogisticRegression(penalty='l1', C=alpha, solver='saga')
        self.regression_clf = linear_model.Lasso(alpha=alpha)
        self.non_scary_score = 0

    def fit(self, x_train, y_train):
        scary_slicer = y_train >= 2
        non_scary = np.array(~scary_slicer, dtype=int)
        self.classification_clf.fit(x_train, non_scary)
        self.regression_clf.fit(x_train, y_train)
        self.non_scary_score = np.mean(y_train[~scary_slicer])

    def predict(self, x_predict):
        y_hat = np.zeros(x_predict.shape[0])
        non_scary_class = self.classification_clf.predict(x_predict)
        y_hat[non_scary_class == 1] = self.non_scary_score
        if np.sum(non_scary_class == 0) > 0:
            scary_y_hat = self.regression_clf.predict(x_predict[non_scary_class == 0, :])
            y_hat[non_scary_class == 0] = scary_y_hat
        return y_hat

    def score(self, x_score, y_score):
        return metrics.r2_score(self.predict(x_score), y_score)



def create_data_mat(csv_path='./set1/mp3/Soundtrack360_mp3/'):

    os.chdir(csv_path)
    csv_files = glob.glob('*.{}'.format('csv'))
    data_mat = []
    for file_name in csv_files:
        print(file_name)
        with open(file_name, 'r') as file:
            features_line = file.readlines()[-1]
            data_mat.append([float(feature) for feature in features_line.split(',')[2:-1]])
    for l in data_mat:
        if len(l) != len(data_mat[0]):
            print(len(l))
    data_mat = np.array(data_mat)
    feature_max = np.max(data_mat, 0).reshape(1, -1)
    feature_min = np.min(data_mat, 0).reshape(1, -1)
    data_mat = (data_mat - feature_min) / (feature_max - feature_min)  # Normalize
    np.savetxt('data_mat.csv', data_mat, delimiter=',')



#create_data_mat()  #Uncomment to create data matrix from individual features csvs
x = np.loadtxt('data_mat.csv', delimiter=',')
y = np.loadtxt('fear_ratings.csv', delimiter=',')

# Compute test error
acc = 0
r2 = 0
low_fear_acc = 0
reps = 100
for i in range(reps):
    inds = np.arange(len(y))
    np.random.shuffle(inds)
    model = MixedModel(0.005)  # try 0.01
    model.fit(x[inds[:300], :], y[inds[:300]])
    y_hat = model.predict(x)
    y_hat = np.maximum(np.minimum(8*np.ones(len(y_hat)), y_hat), np.ones(len(y_hat)))
    acc += np.mean(np.abs(y[inds[300:]] - y_hat[inds[300:]]))
    r2 += metrics.r2_score(y[inds[300:]], y_hat[inds[300:]])

    # Compute accuracy for not scary tracks (<2)
    test_y = y[inds[300:]]
    test_y_hat = y_hat[inds[300:]]
    low_fear_acc += np.mean(test_y_hat[test_y < 2] < 2)

acc /= reps
r2 /= reps
low_fear_acc /= reps

print("Test mean error" + str(acc))
print("Test mean R squared: " + str(r2))
print("Test mean success for not scary: " + str(low_fear_acc))
