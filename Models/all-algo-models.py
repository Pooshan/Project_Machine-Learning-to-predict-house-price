#!/usr/bin/python

from svm import svmModule
from naive_bayes import naiveBayes
from linear_regression import linearRegression
from knn import KNNAlgo
from gradient_descent import gradientDescent
from decision_tree import decisionTree

class allModels:
    def all_algo_model(self):

        print(" \n----------------------------------\n")
        print("\n Decision Tree \n")
        print(" \n----------------------------------\n")

        #decisionTree().decision_tree_algo()


        print(" \n----------------------------------\n")
        print("\n Gradient Descent \n")
        print(" \n----------------------------------\n")

        gradientDescent().gradient_descent_algo()

        print(" \n----------------------------------\n")
        print("\n K-Nearest Neighbour \n")
        print(" \n----------------------------------\n")

        #KNNAlgo().KNN__model_algo()

        print(" \n----------------------------------\n")
        print("\n Linear Regression \n")
        print(" \n----------------------------------\n")

        linearRegression().linear_regression_algo()

        print(" \n----------------------------------\n")
        print("\n Naive Bayes \n")
        print(" \n----------------------------------\n")

        naiveBayes().naive_bayes_algo()

        print(" \n----------------------------------\n")
        print("\n Support Vector Machine \n")
        print(" \n----------------------------------\n")

        svmModule().svm_model_algo()

        print("\n---------- End ---------------------\n")



if __name__ == '__main__':
    x= allModels()
    x.all_algo_model()
