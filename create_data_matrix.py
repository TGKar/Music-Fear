import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import ensemble
from sklearn import ensemble
from sklearn import metrics

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

# Compute test error
def calc_test_error(x, y, alpha):
    mean_err = 0
    r2 = 0
    test_slicer = int(5 * len(y) / 6)
    reps = 10
    for i in range(reps):
        inds = np.arange(len(y))
        np.random.shuffle(inds)
        model = ensemble.RandomForestRegressor(min_samples_split=5)  # try 0.01
        model.fit(x[inds[:test_slicer], :], y[inds[:test_slicer]])
        y_hat = model.predict(x)
        y_hat = np.maximum(np.minimum(8*np.ones(len(y_hat)), y_hat), np.ones(len(y_hat)))
        mean_err += np.mean(np.abs(y[inds[test_slicer:]] - y_hat[inds[test_slicer:]]))
        r2 += metrics.r2_score(y[inds[test_slicer:]], y_hat[inds[test_slicer:]])

    mean_err /= reps
    r2 /= reps

    return mean_err, r2

def calc_training_error(x, y, alpha):
    model = ensemble.RandomForestRegressor(min_samples_split=5, oob_score=True)
    # model = linear_model.LinearRegression()  # try 0.01
    model.fit(x, y)
    y_hat = model.predict(x)
    y_hat = np.maximum(np.minimum(8 * np.ones(len(y_hat)), y_hat), np.ones(len(y_hat)))
    mean_error = np.mean(np.abs(y - y_hat))
    r2 = metrics.r2_score(y, y_hat)
    return mean_error, r2, model.feature_importances_

def plot_feature_importance(importance):
    print("Most important ind " + str(np.argsort(importance)[-5:]))
    plt.bar(np.arange(len(importance)), importance)
    plt.title("Random Forest Variable Importance")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.show()


# create_data_mat()  #Uncomment to create data matrix from individual features csvs
features = np.loadtxt('data_mat.csv', delimiter=',')
fear_rating = np.loadtxt('fear_ratings.csv', delimiter=',')
normalization_const = 0.01
for i in range(10):
    mean_train_error, train_r2, feature_importance = calc_training_error(features, fear_rating,
                                                                    normalization_const)
    print("Most important ind " + str(np.argsort(feature_importance)[-5:]))
print("Train mean error: " + str(mean_train_error))
print("Train mean R squared: " + str(train_r2))

plot_feature_importance(feature_importance)

mean_test_error, test_r2 = calc_test_error(features, fear_rating, normalization_const)
print("Test mean error: " + str(mean_test_error))
print("Test mean R squared: " + str(test_r2))

# feature_names =
# feature_importance =