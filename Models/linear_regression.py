#!/usr/bin/python
import sys
import csv

import matplotlib.pyplot as plt

from sklearn import linear_model, preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# Build X and Y lists
# X : Features
# Y : Target

class linearRegression:

    def linear_regression_algo(self):

        X = []
        Y = []


        with open('../Data/full_table.csv', 'r') as file:
            for line in csv.reader(file, delimiter = ','):
                if len(line) == 13:
                    try:
                        zhvi = float(line[5])
                        property_type = line[6]
                        room_type = line[7]
                        accommodates = int(line[8])
                        bathrooms = float(line[9])
                        beds = int(line[10])
                        bed_type = line[11]
                        price = float(line[12])

                        x = {
                            'zhvi': zhvi,
                            'property_type': property_type,
                            'room_type': room_type,
                            'accommodates': accommodates,
                            'bathrooms': bathrooms,
                            'beds': beds,
                            'bed_type': bed_type
                        }

                        y = price

                        X.append(x)
                        Y.append(y)


                    except:
                        pass


        # The DictVectorizer converts data from a dictionary to an array
        vec = DictVectorizer()

        # Convert X to Array
        X = vec.fit_transform(X).toarray()

        # Split X and Y into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=43)


        # Linear Regression
        model = linear_model.LinearRegression()
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        mse = mean_squared_error(Y_test, Y_pred)
        mae = mean_absolute_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)

        print('Linear Regression')
        print('Mean Squared Error: {0}'.format(mse))
        print('Mean Average Error: {0}'.format(mae))
        print('R2 Score: {0}'.format(r2))

        # With Boosting
        model_boost = AdaBoostRegressor(linear_model.LinearRegression())
        model_boost.fit(X_train, Y_train)
        Y_pred = model_boost.predict(X_test)
        mse = mean_squared_error(Y_test, Y_pred)
        mae = mean_absolute_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)

        print('Linear Regression (with AdaBoost)')
        print('Mean Squared Error: {0}'.format(mse))
        print('Mean Average Error: {0}'.format(mae))
        print('R2 Score: {0}'.format(r2))

        # With Bagging
        model_bag = BaggingRegressor(linear_model.LinearRegression())
        model_bag.fit(X_train, Y_train)
        Y_pred = model_bag.predict(X_test)
        mse = mean_squared_error(Y_test, Y_pred)
        mae = mean_absolute_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)

        print('Linear Regression (with Bagging)')
        print('Mean Squared Error: {0}'.format(mse))
        print('Mean Average Error: {0}'.format(mae))
        print('R2 Score: {0}'.format(r2))
