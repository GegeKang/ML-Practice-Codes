# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler # scaling
from sklearn.preprocessing import OneHotEncoder # encode
from sklearn.preprocessing import LabelEncoder # encode
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer



# Load the Wine Quality Red dataset
#Delimiter: A character that separates columns in a CSV file. Common delimiters include ;, \t and others. comma is default; if implicitly shown, then, seperated by comma.

dataset = pd.read_csv ('Data.csv',delimiter=',')

# Separate features and target
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# take care of missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # create an object of SimpleImputer class
imputer.fit (X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3]) # fill the blank cell with the average of all other data in the same column



# Implement an instance of the ColumnTransformer class
ct = ColumnTransformer (transformers = [('encoder',OneHotEncoder(),[0])], remainder = 'passthrough')
X = np.array (ct.fit_transform (X))# Apply the fit_transform method on the instance of ColumnTransformer; Convert the output into a NumPy array

# Use LabelEncoder to encode binary categorical data
le = LabelEncoder ()
y = le.fit_transform (y)

# Split the dataset into an 80-20 training-test set
X_train,X_test,y_train,y_test = train_test_split (X,y,test_size = 0.2,random_state = 42)

# Create an instance of the StandardScaler class
sc = StandardScaler ()

# Fit the StandardScaler on the features from the training set and transform it
X_train = sc.fit_transform (X_train)

# Apply the transform to the test set
X_test = sc.transform (X_test)

# Print the scaled training and test datasets
print (X_train)
print (X_test)
print (y_train)
print (y_test)