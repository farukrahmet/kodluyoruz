
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection  import train_test_split
from sklearn.model_selection import ShuffleSplit

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import pickle


def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between
        true and predicted values based on the metric chosen. """

    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true ,y_predict)

    # Return the score
    return score


def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a
        decision tree regressor trained on the input data [X, y]. """

    rs = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)
    cv_sets=rs.get_n_splits(X.shape[0])

    # TODO: Create a decision tree regressor object
    regressor = DecisionTreeRegressor(random_state=0)

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth': np.arange(1, 10)}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer'
    scoring_fnc = make_scorer(performance_metric)
    # scoring_fnc = make_scorer(performance_root_mean_square)

    # TODO: Create the grid search cv object --> GridSearchCV()
    # Make sure to include the right parameters in the object:
    grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

def train():
    data = pd.read_csv('housing.csv')
    prices = data['MEDV']
    features = data.drop('MEDV', axis=1)

    # TODO: Shuffle and split the data into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(features,prices, test_size=0.20, random_state=42)

    #find best estimator
    reg = fit_model(X_train, y_train)

    model=reg.fit(X_train,y_train)

    # save file
    filename = "trained_model"
    pickle.dump(model, open(filename, 'wb'))

    # Success
    return "Training  was successful."


def predict(data):
    filename = "trained_model"
    loaded_model = pickle.load(open(filename, 'rb'))
    pred = loaded_model.predict([data])[0]
    result = "Price : ${:,.2f}".format(pred)
    return result


if __name__=='__main__':

    #train model
    # result=train()
    # print(result)

    #predict
    data=[5, 17, 15]
    result=predict(data)
    print(result)
