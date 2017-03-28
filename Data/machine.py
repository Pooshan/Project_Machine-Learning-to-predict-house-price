#!/usr/bin/python
import sys
import csv

from sklearn import linear_model, preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


# Build X and Y lists
# X : Features
# Y : Target
X = []
Y = []


for line in csv.reader(sys.stdin, delimiter = ','):
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


# 1) Set Linear Regression Model
model = linear_model.LinearRegression()

# Uncomment the following code to print the coefficients of the linear regression
#coefficients = vec.inverse_transform(model.coef_)
#print(coefficients)

# 2) Set Stochastic Gradient Descent Model
#model = linear_model.SGDRegressor()

# 3) Set SVM Regression Model
#model = SVR()

# 4) Set Naive Bayes Model
#model = GaussianNB()

# 5) Set Decision Tree Regression Model
#model = DecisionTreeRegressor(max_depth=5)

# 6) Set Nearest Neighbors Regression Model
#model = KNeighborsRegressor(n_neighbors=5)


# Fit training features to training target
model.fit(X_train, Y_train)

# Predict Targets from Test Features
Y_pred = model.predict(X_test)

# Mean Absolute Error
mae = mean_absolute_error(Y_test, Y_pred)
print('Mean Absolute Error: {0}'.format(mae))

# Mean Square Error
mse = mean_squared_error(Y_test, Y_pred)
print('Mean Square Error: {0}'.format(mse))

# R2 Error
r2 = r2_score(Y_test, Y_pred)
print('R2 Error: {0}'.format(r2))
