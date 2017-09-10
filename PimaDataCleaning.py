import pandas as pd
from numpy.random import RandomState
from sklearn import preprocessing

# Read Data
data = pd.read_csv("/Users/yazen/Desktop/datasets/PimaDiabetes/pima.csv")

# Label columns
data.columns = ["pregnancy", "plasma/glucose concentration", "blood pressure","tricep skin fold thickness", "serum insulin", "body mass index", "diabetes pedigree function", "age", "label"]

# Remove rows with missing data
data = data.loc[data["plasma/glucose concentration"] > 20]
data = data.loc[data["blood pressure"] > 60]
data = data.loc[data["body mass index"] > 20]

# Under sample negative rows
negative = data.loc[data["label"] < 1].sample(frac=0.5, random_state=RandomState())
positive = data.loc[data["label"] > 0]
neutral = positive.append(negative)

# Normalize data
min_max_scaler = preprocessing.MinMaxScaler()
neutral = min_max_scaler.fit_transform(neutral)
neutral = pd.DataFrame(neutral,columns = data.columns)

# Create test and training set
train = neutral.sample(frac = .7, random_state=RandomState())
test = neutral.loc[~neutral.index.isin(train.index)]

train.to_csv('/Users/yazen/Desktop/datasets/PimaDiabetes/train.csv')
test.to_csv('/Users/yazen/Desktop/datasets/PimaDiabetes/test.csv')
