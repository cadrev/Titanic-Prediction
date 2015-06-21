# Title  : pandas row shuffler 
# Author : Felan Carlo Garcia 
#
# Notes:
# -- We shuffle the data inside the original
#    Titanic csv file since the data are arranged
#    in order.    

import numpy as np
import pandas as pd

def shuffler(filename):
  df = pd.read_csv(filename, header=0)
  # return the shuffled pandas dataframe
  return df.reindex(np.random.permutation(df.index))

def main(outputfilename):
  # write the shuffled pandas dataframe to a new file
  shuffler('titanic-data.csv').to_csv(outputfilename, sep=',', index=False)

if __name__ == '__main__': 
  main('titanic-data-shuffled.csv')
