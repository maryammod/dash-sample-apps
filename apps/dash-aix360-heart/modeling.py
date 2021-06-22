from aix360.algorithms.rbm import LogisticRuleRegression, FeatureBinarizer
import pandas as pd
from sklearn.model_selection import train_test_split

col_map = {
    "HR": "HR",
    "O2Sat": "O2Sat",
    "Temp": "Temp",
    "SBP": "SBP",
    "MAP": "MAP",
    "DBP": "DBP",
    "Resp": "Resp",
    "Age": "Age",
    "Gender": "Gender",
    "ICULOS": "ICULOS"
}

num2desc = {
    "Gender": {0: "Female", 1: "Male"}
}

# Load and preprocess dataset
df = pd.read_csv("heart.csv")
print(df.head())
for k, v in num2desc.items():
    df[k] = df[k].replace(v)

y = df.pop("target")
dfTrain, dfTest, yTrain, yTest = train_test_split(df, y, random_state=0, stratify=y)

fb = FeatureBinarizer(negations=True, returnOrd=True)
dfTrain, dfTrainStd = fb.fit_transform(dfTrain)
dfTest, dfTestStd = fb.transform(dfTest)

# Train model
lrr = LogisticRuleRegression(lambda0=0.005, lambda1=0.001, useOrd=True)
lrr.fit(dfTrain, yTrain, dfTrainStd)
