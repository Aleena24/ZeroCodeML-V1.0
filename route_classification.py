#flask essentials
from flask import Flask, render_template, request, redirect, session, url_for, Blueprint, jsonify, flash
from flask_login import current_user, login_user, login_required, logout_user, LoginManager

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

#custom import
from classification import *
from classify_visualize import *
from Preprocessing import *
from genAI import *

classification = Blueprint('classification', __name__, url_prefix='/classification')


@classification.route("/<learningType>/<algorithm>/<model_name>/<target_column>", methods = ['POST', 'GET'])
def classify(learningType, algorithm, model_name, target_column):
    print("route classification")

    data = pd.read_csv("data.csv")

    df, encoder, scaler, dropped_columns = preprocess_data(data, 40)
    X_train, X_test, y_train, y_test = split_data_supervised(df, target_column, 70 )

    model, metrics, y_pred = train_classification_model(model_name, X_train, X_test, y_train, y_test)

    
    '''

    Visulaization starts from here

    '''

    '''
    Confusion matrix
    '''
    conm = confusion_matrix(y_test, y_pred)    
    class_name = np.unique(y_test) 

    cm =[]
    cmHighValue = 0
    cmLowValue = 0
    for i in conm:
        mat = []
        for j in i:
            if j > cmHighValue:
                cmHighValue = j
            if j < cmLowValue:
                cmLowValue = j
            mat.append(j)
        cm.append(mat)

    class_names =[]
    for i in class_name:
        class_names.append(i)

    '''
    ROC curve
    '''
    y_probs = model.predict_proba(X_test)
    n_classes = len(model.classes_)
    rocPlot = []

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test, y_probs[:, i], pos_label=i)
        roc_auc = auc(fpr, tpr)
        print(roc_auc)
        rocPlot.append({"name": f"Class {i} (AUC = {roc_auc:.2f})","fpr":list(fpr),"tpr":list(tpr)})


    '''
    Precision and Recall curve
    '''

    prCurve = []

    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_test, y_probs[:, i], pos_label=i)
        prCurve.append({"name": f"Class {i}","precision":list(precision),"recall":list(recall)})

    '''
    GenAI response
    '''
    genAI_response = getDesciption(model_name, metrics)
    # confusion_matrix = plot_confusion_matrix(y_test, y_pred)
    # roc_curve = plot_roc_curve(model,X_test,y_test)
    # precision_recall_curve = plot_precision_recall_curve(model,X_test,y_test)
    # feature_importance = plot_feature_importance(model, X_train)

    return render_template("clas_result.html",
                           metrics = metrics,
                           cm = cm,
                           class_names = class_names,
                           cmHighValue = cmHighValue,
                           cmLowValue = cmLowValue,
                           rocPlot = rocPlot,
                           prCurve = prCurve,
                           genAI_response = genAI_response)




# return redirect(url_for('classification.classificatio_viz',
#                         model_name = model_name,
#                         model = model,
#                         X_train = X_train,
#                         X_test = X_test, 
#                         y_train = y_train, 
#                         y_test=y_test, 
#                         y_pred = y_pred))