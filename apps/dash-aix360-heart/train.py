from interpret.blackbox import LimeTabular
import interpret
from interpret import show
import pickle
from aix360.algorithms.rbm import LogisticRuleRegression
from modeling import dfTrain, dfTrainStd, dfTest, dfTestStd, yTrain, yTest, y
from sklearn.ensemble import RandomForestClassifier
# Train model
#Maryam Since we have 2 different models, Regression classifier used for diagrams 
#and RandomForest for Lime, we should save both models

lrr = LogisticRuleRegression(lambda0=0.005, lambda1=0.001, useOrd=True)
lrr.fit(dfTrain, yTrain, dfTrainStd)
#Maryam save the the trained Regression model into lrr variable
#pickle.dump(lrr, "./saved_regression_model.sav")
pickle.dump(lrr, open("./saved_regression_model.sav", 'wb'))

lrr2 = RandomForestClassifier(n_estimators=100, n_jobs=-1)
lrr2.fit(dfTrain, yTrain)
#Maryam save the the trained RandomForestClassifier model into saved_model variable
#pickle.dump(lrr2, "./saved_randomforest_model.sav")
pickle.dump(lrr2, open("./saved_randomforest_model.sav", 'wb'))
#Blackbox explainers need a predict function, and optionally a dataset

lime_rf = LimeTabular(predict_fn=lrr2.predict_proba, data=dfTrain, random_state=1)
lime = lime_rf.explain_local(dfTest[:2], yTest[:2], name='LIME')
pickle.dump(lime, open("./saved_lime_model.sav", 'wb'))