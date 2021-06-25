from aix360.algorithms.rbm import FeatureBinarizer
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split



col_map = {
    "HR": "HR",
    "O2Sat": "O2Sat",
    "Temp": "Tempreture",
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

for k, v in num2desc.items():
    df[k] = df[k].replace(v)

y = df.pop("target")
dfTrain, dfTest, yTrain, yTest = train_test_split(df, y, random_state=0, stratify=y)

fb = FeatureBinarizer(negations=True, returnOrd=True)
dfTrain, dfTrainStd = fb.fit_transform(dfTrain)
dfTest, dfTestStd = fb.transform(dfTest)

# Train model
# Maryam, transfer training code to the train.app