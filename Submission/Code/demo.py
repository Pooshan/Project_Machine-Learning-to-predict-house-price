#!/usr/bin/python
import sys
import csv

from sklearn.externals import joblib

normalizer = joblib.load('../Models/normalizer.pkl')
vectorizer = joblib.load('../Models/vectorizer.pkl')

linear_regression = joblib.load('../Models/linear_regression.pkl')
gradient_descent = joblib.load('../Models/gradient_descent.pkl')
svm = joblib.load('../Models/svm.pkl')
naive_bayes = joblib.load('../Models/naive_bayes.pkl')
decision_tree = joblib.load('../Models/decision_tree.pkl')
knn = joblib.load('../Models/knn.pkl')

zipcode_rent_table = {}
with open('../Data/full_zipcode_rent.csv') as file:
    for line in csv.reader(file, delimiter=','):
        if len(line) == 2:
            try:
                zipcode, rent = line
                rent = int(rent)
                zipcode_rent_table[zipcode] = rent
            except:
                pass



print('Zipcode, Property Type, Room Type, Accommodates, Bathrooms, Beds, Bed Type, Linear Regression, Gradient Descent, SVM, Naive Bayes, Decision Tree, KNN')
with open('../Data/demo_data.csv') as file:
    for line in csv.reader(file, delimiter=','):
        if len(line) == 7 and line[0] != 'Zipcode':
            try:
                zipcode, property_type, room_type, accommodates, bathrooms, beds, bed_type = line

                average_rent = zipcode_rent_table[zipcode]

                x = {
                    'average_rent': float(average_rent),
                    'property_type': property_type,
                    'room_type': room_type,
                    'accommodates': int(accommodates),
                    'bathrooms': float(bathrooms),
                    'beds': int(beds),
                    'bed_type': bed_type
                }

                x = vectorizer.transform(x).toarray()

                x_norm = normalizer.transform(x)

                linear_regression_pred = linear_regression.predict(x)[0]
                gradient_descent_pred = gradient_descent.predict(x_norm)[0]
                svm_pred = svm.predict(x)[0]
                naive_bayes_pred = naive_bayes.predict(x)[0]
                decision_tree_pred = decision_tree.predict(x)[0]
                knn_pred = knn.predict(x)[0]

                output = ','.join([
                    zipcode,
                    property_type,
                    room_type,
                    accommodates,
                    bathrooms,
                    beds,
                    bed_type,
                    str(linear_regression_pred),
                    str(gradient_descent_pred),
                    str(svm_pred),
                    str(naive_bayes_pred),
                    str(decision_tree_pred),
                    str(knn_pred)
                ])

                print(output)


            except:
               pass
