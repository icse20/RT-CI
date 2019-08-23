from sklearn.model_selection import train_test_split
import xml.etree.cElementTree as ET
import understand
import sys
import subprocess
import shlex
import operator
import numpy as np
import csv
import collections
import pandas as pd
import os
import random
import time
from datetime import datetime
from sklearn import neural_network
try:
	import cPickle as pickle
except:
	import pickle
    


    
if __name__ == '__main__':
    data = pd.read_csv('commons_imaging.csv', header = 0)
    data = data.drop('errors', axis = 'columns')
    data = data.drop('errors_%', axis = 'columns')
    data.to_csv('commons_imaging_new.csv', index = False)