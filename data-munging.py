# 
# Title  : Data munging(AKA cleaning) the Titanic Data 
# Author : Felan Carlo Garcia
#
# Notes:
#  -- Code is based on the Kaggle Python Tutorial
#  -- data cleaning prior to implementing a machine learning algorithm.


import numpy as np 
import pandas as pd 

def processdata(filename, outputname):

  df = pd.read_csv(filename,header=0)
  
  # Make a new column 'Gender' and EmbarkedNum to convert the string
  # information into an integer value.
  # We do this because general machine learning algorithms do not
  # work on string values.
  df['Gender']      = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
  df['EmbarkedNum'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 1}).astype(int)
 
  # Executing the code:
  # --print df[df['Age'].isnull()][Sex']-- shows that the titanic data contains
  # some null values of the ages of the passengers. 
  # In this case, we can either drop the row or we can assign an arbitrary 
  # value to fill the missing data.
  # For this code, arbitrary age data is obtained by using the median 
  # age data of the passengers. We make a new column 'AgeFill' and place
  # the median data on the missing values instead of directly modifying 
  # the 'Age' column

  df['AgeFill'] = df['Age']
  for i in range(0, 2):
    for j in range(0, 3):
      median = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()
      df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),'AgeFill'] = median
      
  # We add a new column 'AgeIsNull' to know which records has a missing
  # values previously. 
  # We then interpolate the missing values from the 'Fare' column.
  df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
  df['Fare']      = df['Fare'].interpolate()

  # ------------- Feature Engineering Part --------------------
  # Feature Engineering is the process of using domain/expert 
  # knowledge of the data to create features that make machine
  # learning algorithms work better. 
  #
  # In this case, studying the data shows that women and children
  # have higher survival rates compared to men. Thus we add 
  # two additional features: 'Female' and 'Children', in an attempt
  # to assist our learning model in its prediction.
  # At the same time we add features Age*Class and FamilySize
  # as additional engineered feature that may help our learning 
  # model
  df['Children']   = df['AgeFill'].map(lambda x: 1 if x < 6.0 else 0)
  df['Female']     = df['Gender'].map(lambda x: 1 if x == 0 else 0)
  df['FamilySize'] = df['SibSp']   + df['Parch']
  df['Age*Class']  = df['AgeFill'] * df['Pclass']


  # Since most machine learning algorithms don't work on strings,
  # we drop the columns in our pandas dataframe containing object
  # datatypes.
  # The code:
  # --print df.dtypes[df.dtypes.map(lambda x: x=='object')]--
  # will show which columns are made of object datatypes.
  #
  # In this case these are the following columns containing 
  # object.string:
  # Age, Name, Sex, Ticket, Cabin, Embarked, Fare
  #
  # We drop the following objects columns along with the other data
  # since they wont likely contribute to our machine learning 
  # prediction
  df = df.drop(['Age','Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
  df.to_csv(outputname, sep=',', index=False)

  return df


def main():
  print processdata('titanic-data-shuffled.csv', 'final-data.csv')



if __name__ == '__main__': 
  main()
