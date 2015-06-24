#
# Title  : Basic Random Forest Prediction with
#          optimized parameters using GridSearchCV
#
# Author : Felan Carlo C. Garcia
#

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# These are the training/testing attributes for the Random Forest Classifier 
attrib = ['Pclass','SibSp','Gender','Parch','EmbarkedNum','AgeFill', 'AgeIsNull', 'FamilySize', 'Age*Class', 'Female', 'Children','Fare']

# Load the fina data set.
data   = pd.read_csv('final-data.csv', header=0)

# Since final-data.csv data contains the whole
# shuffled titanic data, we used 2/3 of 
# the data for training and 1/3 for testing
train  = data[0:872]
test   = data[872::]

# Train the random forest and used the parameters found using GridSearchCV
model  = RandomForestClassifier(n_estimators=65, criterion='gini', max_depth=5, max_features=0.5,random_state=600)
model.fit(train[attrib],train['Survived'])

# Predict data and store on final .csv file.\
test['Predicted'] = model.predict(test[attrib])
test.to_csv('survival-prediction.csv', sep=',', index=False)

# Output result: 0.828
print model.score(test[attrib], test['Survived']) 


