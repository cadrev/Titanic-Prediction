#
# Title  : GridSearchCV parameter search
# Author : Felan Carlo C. Garcia
# Notes
# -- This script is used to determine the optimized parameters for
#    the RandomForestClassifier in order to provide better predictions

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

# GridSearchCV parameters
search_param = {
    "n_estimators": [100],
    "criterion": ["gini", "entropy"],
    'max_features': [0.5, 1.0, "sqrt"],
    'max_depth': [4, 5, 6, 7, 8, None],
}

# These are the training/testing attributes for the Random Forest Classifier 
attrib = ['Pclass','SibSp','Gender','Parch','EmbarkedNum','AgeFill', 'AgeIsNull', 'FamilySize', 'Age*Class', 'Female', 'Children','Fare']
# Load the fina data set.
data   = pd.read_csv('final-data.csv', header=0)


# Since test-final.csv contains the whole
# shuffled titanic data, we used 2/3 of 
# the data for training and 1/3 for testing
train = data[0:872]
test  = data[872::]


# We place a RandomForestClassifier inside a Grid Search
# in order to find the optimal parameters for an optimized
# prediction
forest = RandomForestClassifier(random_state=321)
grid   = GridSearchCV(forest, search_param, cv=12, verbose=0)
grid.fit(train[attrib], train['Survived'])

print grid.best_score_
print grid.best_params_

# -----------------------------------------------------------------------------------
# GridSearch Sample Output: 
# {n_estimators=65, criterion='gini', max_depth=5, max_features=0.5,random_state=321}
# This best output is then placed on a RandomForestClassifier.
# -----------------------------------------------------------------------------------
