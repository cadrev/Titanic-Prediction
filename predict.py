#
# Title  : Basic Random Forest Prediction 
# Author : Felan Carlo C. Garcia
#

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the fina data set.
data   = pd.read_csv('final-data.csv' , header=0)

# We initialize a Random Forest with 1000 trees 
forest = RandomForestClassifier(n_estimators=1000)

# These are the training/testing attributes needed by the Random Forest Classifier 
attrib = ['Pclass','SibSp','Gender','Parch','EmbarkedNum','AgeFill', 'AgeIsNull', 'FamilySize', 'Age*Class']

# Since test-final.csv contains the whole
# shuffled titanic data, we used 2/3 of 
# the data for training and 1/3 for testing
train = data[0:872]
test  = data[872::]

# Train the random forest using the training data
# and make a new column to store the predicted data.
# We can determine the the accuracy of our prediction
# using the forest.score() method 
forest.fit(train[attrib],train['Survived'])
test['Predict'] = forest.predict(test[attrib])
predict_score   = forest.score(test[attrib],test['Survived'])

# Store the predicted values on a csv file
test[['Survived', 'Predict']].to_csv('prediction-output.csv', sep=',')

print predict_score
print test[['Survived', 'Predict']]


