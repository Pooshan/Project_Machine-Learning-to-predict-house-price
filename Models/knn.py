#!/usr/bin/python
import sys
import csv

import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor


# Build X and Y lists
# X : Features
# Y : Target

class KNNAlgo:

    def KNN__model_algo(self):

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
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


        mse_array = []
        mae_array = []
        r2_array = []

        mse_boost_array = []
        mae_boost_array = []
        r2_boost_array = []

        mse_bag_array = []
        mae_bag_array = []
        r2_bag_array = []

        n_array = [ n + 1 for n in range(50)]
        for n in n_array:
            # KNN
            model = KNeighborsRegressor(n_neighbors=n)
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)
            mse = mean_squared_error(Y_test, Y_pred)
            mse_array.append(mse)
            mae = mean_absolute_error(Y_test, Y_pred)
            mae_array.append(mae)
            r2 = r2_score(Y_test, Y_pred)
            r2_array.append(r2)

            # With Boosting
            model_boost = AdaBoostRegressor(KNeighborsRegressor(n_neighbors=n))
            model_boost.fit(X_train, Y_train)
            Y_pred = model_boost.predict(X_test)
            mse = mean_squared_error(Y_test, Y_pred)
            mse_boost_array.append(mse)
            mae = mean_absolute_error(Y_test, Y_pred)
            mae_boost_array.append(mae)
            r2 = r2_score(Y_test, Y_pred)
            r2_boost_array.append(r2)

            # With Bagging
            model_bag = BaggingRegressor(KNeighborsRegressor(n_neighbors=n))
            model_bag.fit(X_train, Y_train)
            Y_pred = model_bag.predict(X_test)
            mse = mean_squared_error(Y_test, Y_pred)
            mse_bag_array.append(mse)
            mae = mean_absolute_error(Y_test, Y_pred)
            mae_bag_array.append(mae)
            r2 = r2_score(Y_test, Y_pred)
            r2_bag_array.append(r2)

            print('Completed for {0} neighbors'.format(n))

        plt.plot(n_array, mse_array, 'b-', label='MSE')
        plt.plot(n_array, mse_boost_array, 'r-', label="MSE with AdaBoost")
        plt.plot(n_array, mse_bag_array, 'g-', label='MSE with Bagging')
        plt.xlabel('Neighbors')
        plt.ylabel('Mean Squared Error')
        plt.show()

        plt.plot(n_array, mae_array, 'b-', label='MAE')
        plt.plot(n_array, mae_boost_array, 'r-', label="MAE with AdaBoost")
        plt.plot(n_array, mae_bag_array, 'g-', label='MAE with Bagging')
        plt.xlabel('Neighbors')
        plt.ylabel('Mean Absolute Error')
        plt.show()

        plt.plot(n_array, r2_array, 'b-', label='R2')
        plt.plot(n_array, r2_boost_array, 'r-', label="R2 with AdaBoost")
        plt.plot(n_array, r2_bag_array, 'g-', label='R2 with Bagging')
        plt.xlabel('Neighbors')
        plt.ylabel('R2 Score')
        plt.show()
