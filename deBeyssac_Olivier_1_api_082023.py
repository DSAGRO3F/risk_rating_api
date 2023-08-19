
import os
import flask
from flask import Flask, jsonify, request, render_template, redirect, url_for, app, redirect
import pandas as pd
import numpy as np
from pip._vendor.rich import table
import html
import pickle
import sklearn
import json

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/Hello')
def api_welcome():
    return "<h1>Welcome to home credit application<h1><p>This application is for evaluating client eligibility to credit<p>"


# Set destination files we need
#dir_path = '/Users/olivierdebeyssac/Open_Classrooms/Data_scientist/Projet_7/Data'
dir_path = 'Data'

file_name_X_test = 'X_test.csv'
destination_path_X_test = dir_path + '/' + file_name_X_test


file_name_X_train = 'X_train.csv'
destination_path_X_train = dir_path + '/' + file_name_X_train


file_name_y_train = 'y_train.csv'
destination_path_y_train = dir_path + '/' + file_name_y_train


# Set api to serve X_test dataset (for the user to be able to choose client)
@app.route('/get_X_test_data', methods=['GET', 'POST'])
def get_test_data_set():
    X_test = pd.read_csv(destination_path_X_test, sep=',')
    print('X_test shape__: {}'.format(X_test.shape))
    id = X_test.loc[:, ['SK_ID_CURR']]
    id = id.astype('str')
    id = id.values.tolist()
    id = [element for subelement in id for element in subelement]
    id = json.dumps(id)

    return id

#print(type(get_test_data_set()))
#print(get_test_data_set())



@app.route('/get_input_data/<customer_id>', methods=['GET', 'POST'])
def file(customer_id):
    print("Customer_id: {}".format(customer_id))
    df_input = pd.read_csv(destination_path_X_test, sep=',')
    #print(df_input.shape)
    #print(df_input.head(5))

    # As customer_id is a string (ex: 100280), convert df_input['SK_ID_CURR'] as string object
    df_input['SK_ID_CURR'] = df_input['SK_ID_CURR'].astype('int')
    df_input['SK_ID_CURR'] = df_input['SK_ID_CURR'].astype('str')

    # Search customer_id id df_input()
    df_client = df_input.loc[df_input['SK_ID_CURR'] == customer_id]
    df_client_cols = df_client.columns
    # print(df_client)
    # print(df_client.shape)
    # print(type(customer_id))
    # print(customer_id)

    # Read model object
    #Models = '/Users/olivierdebeyssac/Open_Classrooms/Data_scientist/Projet_7/Livrables/Models'
    Models = 'Models'
    model = Models + '/' + 'model.pkl'
    fichier = open(model, 'rb')
    rf = pickle.load(fichier)

    # Make prediction
    y_pred = rf.predict(df_client)
    print("Prediction: {}".format(y_pred))

    # Select relevant client features
    clt_info = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
                'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH',
                'DAYS_EMPLOYED', 'OWN_CAR_AGE', 'CNT_FAM_MEMBERS', 'EXT_SOURCE_2',
                'EXT_SOURCE_3', 'NAME_FAMILY_STATUS_Married','REG_CITY_NOT_WORK_CITY',
                'DAYS_EMPLOYED_PERC', 'INCOME_CREDIT_PERC', 'ANNUITY_INCOME_PERC', 'PAYMENT_RATE']
    df_clt_info = df_client.loc[:, clt_info]

    # Wrap objects
    wrap = {'customer_id':customer_id, 'y_pred': y_pred.tolist(),
            'df_client': df_client.to_json(), 'df_clt_feat': df_clt_info.to_json()}

    return jsonify(wrap)


@app.route('/feat_imp/<customer_id>',methods=['GET'])
def feat_imp(customer_id):
    # Load random forest model. We need that step to get feature importance.
    #Models = '/Users/olivierdebeyssac/Open_Classrooms/Data_scientist/Projet_7/Livrables/Models'
    Models = 'Models'
    model = Models + '/' + 'model.pkl'
    fichier = open(model, 'rb')
    rf = pickle.load(fichier)

    # Get column names in order to create dataframe
    df_input = pd.read_csv(destination_path_X_test, sep=',')
    print('df_input___{}'.format(df_input[0:2]))

    # If SK_ID_CURR not in proper format...
    # df_input['SK_ID_CURR'] = df_input['SK_ID_CURR'].astype('int')
    df_input['SK_ID_CURR'] = df_input['SK_ID_CURR'].astype('str')

    # Get feature importance from best_estimator_ method (the one with best parameters)
    #rf_best_estimator = rf.best_estimator_
    f_imp = rf.feature_importances_
    #print(f_imp.shape)

    # Build dataframe and keep the five first features
    df_f_imp = pd.DataFrame(f_imp, columns=['feat_imp_value'], index=df_input.columns)
    df_f_imp = df_f_imp.sort_values(by='feat_imp_value', axis=0, ascending=False)
    df_f_imp.reset_index(inplace=True)

    #f_imp_cols = df_f_imp['feat_imp'].columns
    df_f_imp.rename(columns={'index': 'feat_imp'}, inplace=True)

    # Number of most important features to consider.
    f_imp = df_f_imp[0:10]
    #print("f_imp: {}".format(f_imp))
    cols = f_imp['feat_imp'].tolist()
    f_imp = f_imp.to_dict()

    # From train set (limited to a sample of 800 clients) read feature importances values for true positive clients (target==1)
    n_rows = 800
    df_X_train = pd.read_csv(destination_path_X_train, sep=',', nrows=n_rows)
    df_y_train = pd.read_csv(destination_path_y_train, sep=',', nrows=n_rows)
    df = pd.concat([df_X_train, df_y_train], axis=1)
    df_eligible = df[df['TARGET'] == 1]

    # From df_eligible, build new df with feat_imp only
    df_elig_f_imp = df_eligible.loc[:, cols]
    #print("df_elig_f_imp: {}".format(df_elig_f_imp))

    # Do same thing for current file from test set
    print('CUSTOMER_ID: {}'.format(customer_id))
    df_client = df_input.loc[df_input['SK_ID_CURR'] == customer_id]

    #print('df_client: {}'.format(df_client))
    df_client_f_imp = df_client.loc[:,cols]
    #print("df_clientf_imp {}".format(df_client_f_imp))


    # wrap objects
    wrap_f_imp = {'feature_importances': f_imp, 'eligible_clients': df_elig_f_imp.to_json(),
            'NOT_eligible_client': df_client_f_imp.to_json()}

    return jsonify(wrap_f_imp)




#app.run()

