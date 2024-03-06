import pandas as pd
from sklearn import metrics
from classify_helper import * 
import streamlit as st
import pickle

def classification_models(x_train, y_train, x_test, y_test):
    LogisticRegression = logreg(x_train,y_train)
    DecisionTreeClassifier = decisionTree(x_train,y_train)
    RandomForestClassifier = randomForest(x_train, y_train)
    GradientBoostingClassifier = gradientBoost(x_train, y_train)
    AdaBoostClassifier = adaBoost(x_train, y_train)
    KNeighborsClassifier = knn(x_train, y_train)
    
    # LogisticRegression.fit(x_train,y_train)
    # DecisionTreeClassifier.fit(x_train,y_train)
    # RandomForestClassifier.fit(x_train,y_train)
    # GradientBoostingClassifier.fit(x_train,y_train)
    # AdaBoostClassifier.fit(x_train,y_train)
    # KNeighborsClassifier.fit(x_train,y_train)
    
    result = {'Models' : [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}
    
    # LOGISTIC REGRESSION
    y_pred = LogisticRegression.predict(x_test)
    log_report = metrics.classification_report(y_test, y_pred, output_dict = True)

    result['Models'].append("LogisticRegression") 
    result['Accuracy'].append(log_report['accuracy'])
    result['Precision'].append(log_report['macro avg']['precision'])
    result['Recall'].append(log_report['macro avg']['recall'])
    result['F1 Score'].append(log_report['macro avg']['f1-score'])


   # DECISION TREE CLASSIFIER
    y_pred = DecisionTreeClassifier.predict(x_test)
    dt_report = metrics.classification_report(y_test, y_pred, output_dict = True)
    
    result['Models'].append("DecisionTreeClassifier")
    result['Accuracy'].append(dt_report['accuracy'])
    result['Precision'].append(dt_report['macro avg']['precision'])
    result['Recall'].append(dt_report['macro avg']['recall'])
    result['F1 Score'].append(dt_report['macro avg']['f1-score'])
    
    # RANDOM FOREST CLASSIFIER
    y_pred = RandomForestClassifier.predict(x_test)
    rf_report = metrics.classification_report(y_test, y_pred, output_dict = True)
    
    result['Models'].append("RandomForestClassifier") 
    result['Accuracy'].append(rf_report['accuracy'])
    result['Precision'].append(rf_report['macro avg']['precision'])
    result['Recall'].append(rf_report['macro avg']['recall'])
    result['F1 Score'].append(rf_report['macro avg']['f1-score'])

    
    #GRADIENT BOOSTING CLASSIFIER
    y_pred = GradientBoostingClassifier.predict(x_test)
    gb_report = metrics.classification_report(y_test, y_pred, output_dict = True)
    
    result['Models'].append("GradientBoostClassifier") 
    result['Accuracy'].append(gb_report['accuracy'])
    result['Precision'].append(gb_report['macro avg']['precision'])
    result['Recall'].append(gb_report['macro avg']['recall'])
    result['F1 Score'].append(gb_report['macro avg']['f1-score'])

    #ADABOOST CLASSIFIER
    y_pred = AdaBoostClassifier.predict(x_test)
    ada_report = metrics.classification_report(y_test, y_pred, output_dict = True)
    
    result['Models'].append("AdaBoostClassifier") 
    result['Accuracy'].append(ada_report['accuracy'])
    result['Precision'].append(ada_report['macro avg']['precision'])
    result['Recall'].append(ada_report['macro avg']['recall'])
    result['F1 Score'].append(ada_report['macro avg']['f1-score'])

    #KNEIGHBOURS CLASSIFIER    
    y_pred = KNeighborsClassifier.predict(x_test)
    knn_report = metrics.classification_report(y_test, y_pred, output_dict = True)
    
    result['Models'].append("KNeighborsClassifier")
    result['Accuracy'].append(knn_report['accuracy'])
    result['Precision'].append(knn_report['macro avg']['precision'])
    result['Recall'].append(knn_report['macro avg']['recall'])
    result['F1 Score'].append(knn_report['macro avg']['f1-score'])

#Converting the DataFrame
    
    result = dict(sorted(result.items(), key=lambda item: item[1]))
    
    result_df = pd.DataFrame(result, index = None)
    
    st.write(result_df)
    
    #Pickle files
    pickles = {}
    
    pickle.dump(LogisticRegression, open('pickles\LogisticRegression.pkl', 'wb'))
    pickle.dump(DecisionTreeClassifier, open('pickles\DecisionTreeClassifier.pkl', 'wb'))
    pickle.dump(RandomForestClassifier, open('pickles\RandomForestClassifier.pkl', 'wb'))
    pickle.dump(GradientBoostingClassifier, open('pickles\GradientBoostingClassifier.pkl', 'wb'))
    pickle.dump(AdaBoostClassifier, open('pickles\AdaBoostClassifier.pkl', 'wb'))
    pickle.dump(KNeighborsClassifier, open('pickles\KNeighborsClassifier.pkl', 'wb'))
    
    pickles['LogisticRegression'] = 'pickles\LogisticRegression.pkl'
    pickles['DecisionTreeClassifier'] = 'pickles\DecisionTreeClassifier.pkl'
    pickles['RandomForestClassifier'] = 'pickles\RandomForestClassifier.pkl'
    pickles['GradientBoostingClassifier'] = 'pickles\GradientBoostingClassifier.pkl'
    pickles['AdaBoostClassifier'] = 'pickles\AdaBoostClassifier.pkl'
    pickles['KNeighborsClassifier'] = 'pickles\KNeighborsClassifier.pkl'
    
    return pickles
   