import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, auc, roc_curve
import pickle

class RFClassifier:
  def __init__(self,models_path,estimators_n,jobs_n,average='binary'):
    self.models_path = models_path
    self.rf_classifier = RandomForestClassifier(n_estimators=estimators_n,  n_jobs=jobs_n,random_state=42)
    self.average=average
  def multiple_training(self,estimators_n,jobs_n, x_train, y_train, x_test, y_test):
    test_score_RFC=[]
    RFCs=[]
    for n in estimators_n:
        clf = RandomForestClassifier(n_estimators= int(n), n_jobs= jobs_n,random_state=42)
        clf.fit(x_train, np.ravel(y_train))
        y_pred = clf.predict(x_test)
        scores = self.scores(np.ravel(y_test),np.ravel(y_pred))
        test_score_RFC.append(scores)  
        RFCs.append(clf)
    for neighbor, tr_sc in zip((estimators_n),test_score_RFC): 
        print(f"Estimator = {neighbor}")
        print('Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'.format(tr_sc[0], tr_sc[1] ,tr_sc[2], tr_sc[3]))
    return RFCs

  def tain(self, x_train, y_train):
    # Train a Random Forest classifier on the training set
    self.rf_classifier.fit(x_train, np.ravel(y_train))
     
  def predict(self, x_test):
    # Evaluate the performance of the classifier on the testing set
    y_pred = self.rf_classifier.predict(x_test)
    return y_pred

  def scores(self, y_test, y_pred):     
    prec = precision_score(y_test, y_pred,average=self.average)
    rec = recall_score(y_test, y_pred,average=self.average)
    f1 = f1_score(y_test, y_pred,average=self.average)
    accuracy = accuracy_score(y_test, y_pred)
    return [accuracy,prec,rec,f1]

  def print_scores(self,acc,prec,rec,f1):
    print("Accuracy: {}".format(acc))
    print("Precision: {}".format(prec))
    print("Recall: {}".format(rec))
    print("F1: {}".format(f1))
     
  def save_model(self, file_name):
    # save the model
    with open(self.models_path+file_name, 'wb') as f:
        pickle.dump(self.rf_classifier, f)

  def load_model(self,file_name):
    # Load the model                       
    f = open(self.models_path+file_name, 'rb')
    self.rf_classifier = pickle.load(f)   

    #with open(self.models_path+file_name) as f:
     #   self.rf_classifier = pickle.load(f)
    return self.rf_classifier