import pandas as pd
import numpy as np
import sklearn as sk


#reading the data file
file = open("bigmartsales.csv", "r")
content = file.read()
print(content)
file.close()