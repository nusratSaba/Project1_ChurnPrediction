import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


#companyLogo = Image.open("Data/logoJPG.jpg")
#st.set_page_config(page_title='Churn Prediction by TechTrioZ', page_icon=companyLogo)
st.title('Churn Prediction')

option = st.sidebar.selectbox("Select one:", ("Documentation", "Train a dataset"))
if option == "Documentation":
    from documentation import generalDescription
    generalDescription()

if option == "Train a dataset":
    from training import fileUpload
    train_df = fileUpload('Upload the training dataset') # uploading the train dataset

    if train_df is not None:
        from training import Analytics, CustomizeCleaning
        train = Analytics() # an obj of analytics class
        train.targetVisualization(train_df) # target overview

        feature_df = train_df.iloc[:, 1:-1]  # all features except customer ID and target column

        # Visualization of features vs target
        chooseForHist = st.sidebar.multiselect('Choose a feature to visualize it', (feature_df.columns))
        for col in chooseForHist:
            train.histogram(train_df,col)

        # Customize df
        chooseTORemove = st.sidebar.multiselect("Choose the features you don't want to remove", (feature_df.columns))
        custom_df = train_df
        for col in chooseTORemove:
            custom_df = custom_df.drop(col, axis=1)

        st.write('Customize Dataset',custom_df.head())

        #Customize data type and null check
        cleanTrain = CustomizeCleaning()
        cleanTrain.typeCheck(custom_df)

        cleanCheck = st.sidebar.checkbox("Customize data cleaning")
        if cleanCheck:
            custom_df1 = cleanTrain.typeChange(custom_df) #returned
            cleanTrain.typeCheck(custom_df1) # typeChanged

            #cleanTrain.encode(custom_df1) # encode df
            encoded_df = cleanTrain.encode(custom_df1) #returned

            outlierCheck = st.sidebar.checkbox("Check Outliers")
            if outlierCheck:
                encoded_df = cleanTrain.outlierCheck(encoded_df)
                #st.write(encoded_df)

            if encoded_df is not None:
                # Customize Splitting
                #noOfColsEncode = len(encoded_df.axes[1])
                #noOfColsEncode = len(encoded_df.columns)
                listofCol = encoded_df.axes[1]
                noOfColsEncode = len(listofCol)

                X = encoded_df.iloc[:, 1:noOfColsEncode - 1]
                y = encoded_df.iloc[:, noOfColsEncode - 1:noOfColsEncode]

                testSize = st.sidebar.slider("Modify the test size", min_value=0.1, max_value=0.5, value=0.2)
                st.sidebar.write(f"Train size : {1 - testSize}")

                X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=testSize,
                                                                    random_state=42)
                trainSize = X_train.shape
                trainRow = trainSize[0]

                # ML model implementation
                modelName = st.radio("Choose a model",
                                options=['KNN', 'SVM', 'Decision Tree', 'Random Forest Classifier', 'ANN'])

                # KNN -- k
                if modelName == 'KNN':
                    from sklearn.neighbors import KNeighborsClassifier

                    k = round((math.sqrt(trainRow)), 0)
                    k_min = round(k - 10, 0)
                    k_max = round(k + 10, 0)
                    k_parameter = st.number_input("Modify the value of K", min_value=int(k_min), max_value=int(k_max),
                                              value=int(k))
                    # st.write(k,k_max,k_min)

                    knn = KNeighborsClassifier(k_parameter)
                    knn.fit(X_train, y_train)

                    # Make Prediction
                    y_train_pred = knn.predict(X_train)
                    y_test_pred = knn.predict(X_test)

                    # Training set performance
                    knn_train_accuracy = accuracy_score(y_train, y_train_pred)

                    # Test set performance
                    knn_test_accuracy = accuracy_score(y_test, y_test_pred)

                    # st.write('Model performance for taining set')
                    # st.write('-Accuracy: %s' % knn_train_accuracy)

                    # st.write('Model performance for test set')
                    st.write('-Accuracy: %s' % knn_test_accuracy)

                    # saving the model
                    filename = 'final_model.sav'
                    joblib.dump(knn, filename)

                # SVM -- gamma,c
                if modelName == 'SVM':
                    from sklearn.svm import SVC

                    customize_C = st.number_input("Modify the value of C", value=1)
                    customize_gamma = st.number_input("Modify the value of gamma", value=3)

                    svm_rbf = SVC(gamma=customize_gamma, C=customize_C)
                    svm_rbf.fit(X_train, y_train)

                    # Make predictions
                    y_train_pred = svm_rbf.predict(X_train)
                    y_test_pred = svm_rbf.predict(X_test)

                    # Training set performance
                    svm_rbf_train_accuracy = accuracy_score(y_train, y_train_pred)  # Calculate Accuracy

                    # Test set performance
                    svm_rbf_test_accuracy = accuracy_score(y_test, y_test_pred)  # Calculate Accuracy

                    # st.write('Model performance for Training set')
                    # st.write('- Accuracy: %s' % svm_rbf_train_accuracy)

                    # st.write('Model performance for Test set')
                    st.write('- Accuracy: %s' % svm_rbf_test_accuracy)

                    # saving the model
                    filename = 'final_model.sav'
                    joblib.dump(svm_rbf, filename)

                # Decision Tree -- maxDepth
                if modelName == 'Decision Tree':
                    from sklearn.tree import DecisionTreeClassifier

                    customize_depth = st.number_input("Modify the value maximum depth of the tree", value=3)

                    dt = DecisionTreeClassifier(max_depth=customize_depth)  # Define classifier
                    dt.fit(X_train, y_train)  # Train model

                    # Make predictions
                    y_train_pred = dt.predict(X_train)
                    y_test_pred = dt.predict(X_test)

                    # Training set performance
                    dt_train_accuracy = accuracy_score(y_train, y_train_pred)  # Calculate Accuracy

                    # Test set performance
                    dt_test_accuracy = accuracy_score(y_test, y_test_pred)  # Calculate Accuracy

                    # st.write('Model performance for Training set')
                    # st.write('- Accuracy: %s' % dt_train_accuracy)

                    # st.write('Model performance for Test set')
                    st.write('- Accuracy: %s' % dt_test_accuracy)

                    # saving the model
                    filename = 'final_model.sav'
                    joblib.dump(dt, filename)

                # RandomForestClassifier -- n_estimators=10
                if modelName == 'Random Forest Classifier':
                    from sklearn.ensemble import RandomForestClassifier

                    customize_nEstimator = st.number_input("Modify the value of n_estimators", value=10)

                    rf = RandomForestClassifier(n_estimators=customize_nEstimator)  # Define classifier
                    rf.fit(X_train, y_train)  # Train model

                    # Make predictions
                    y_train_pred = rf.predict(X_train)
                    y_test_pred = rf.predict(X_test)

                    # Training set performance
                    rf_train_accuracy = accuracy_score(y_train, y_train_pred)  # Calculate Accuracy

                    # Test set performance
                    rf_test_accuracy = accuracy_score(y_test, y_test_pred)  # Calculate Accuracy

                    # st.write('Model performance for Training set')
                    # st.write('- Accuracy: %s' % rf_train_accuracy)

                    # st.write('Model performance for Test set')
                    st.write('- Accuracy: %s' % rf_test_accuracy)

                    # saving the model
                    filename = 'final_model.sav'
                    joblib.dump(rf, filename)

                if modelName == 'ANN':
                    from sklearn.neural_network import MLPClassifier

                    customize_epochs = st.number_input("Modify the number of epochs", min_value=10, max_value=1000,
                                                   value=50)

                    mlp = MLPClassifier(alpha=1, max_iter=customize_epochs)
                    mlp.fit(X_train, y_train)

                    # Make predictions
                    y_train_pred = mlp.predict(X_train)
                    y_test_pred = mlp.predict(X_test)

                    # Training set performance
                    mlp_train_accuracy = accuracy_score(y_train, y_train_pred)  # Calculate Accuracy

                    # Test set performance
                    mlp_test_accuracy = accuracy_score(y_test, y_test_pred)  # Calculate Accuracy

                    # st.write('Model performance for Training set')
                    # st.write('- Accuracy: %s' % mlp_train_accuracy)

                    # st.write('Model performance for Test set')
                    st.write('- Accuracy: %s' % mlp_test_accuracy)

                    # saving the model
                    filename = 'final_model.sav'
                    joblib.dump(mlp, filename)

                    # st.write(noOfColsEncode)

                do_predict = st.sidebar.checkbox('Want to predict')
                if do_predict:
                    wayToPredict = st.sidebar.selectbox("How would you like to predict?", ("Batch", "Online"))
                    if wayToPredict == "Online":
                        st.info("For A Single Customer")
                        # st.write(encode_df1) #encoded df
                        # st.write(clean_df1) # Final customized df
                        online_df = custom_df.iloc[:, 1:-1]
                        # st.write(online_df)
                        online_dict = {}
                        for col in online_df:
                            # st.write(col)
                            if len(pd.unique(online_df[col])) <= 50:
                                uniqueValue_list = (online_df[col].unique()).tolist()
                                colInput = st.selectbox(f'Choose for {col}', (uniqueValue_list))

                            # uniqueColList = uniqueValueSearch(col) #use this to map
                            # st.write(uniqueColList)
                            else:
                                if online_df.dtypes[col] == 'O':
                                    online_df[col] = pd.to_numeric(online_df[col], errors='coerce')
                                maxRange = int(max(online_df[col]))
                                minRange = int(min(online_df[col]))
                                colInput = st.number_input(f"Give input for {col}", min_value=minRange, max_value=maxRange)

                            online_dict[col] = colInput  # all inputs in a dict

                        # st.write(online_dict) #testing
                        # accessing the dict values (user inputs)
                        online_value = list(online_dict.values())


                        # st.write(online_value)

                        def transformEncode(df):
                            if df[col].nunique() >= 100 and df.dtypes[col] == 'O':
                                df[col] = pd.to_numeric(df[col], errors='coerce')

                            return (df)


                        Transform_DF = transformEncode(online_df)


                        def encodeInputs(col):
                            counter = 0
                            transform_dict = {}
                            listUnique = Transform_DF[col].unique()

                            for len_listUnique in range(len(listUnique)):
                                transform_dict[listUnique[counter]] = counter
                                counter = counter + 1
                            # st.write(transform_dict)
                            return (transform_dict)


                        # encodeInputs('gender')
                        # encodeInputs('Partner')
                        # st.write('actual input',online_value) #testing

                        count = 0
                        for col in online_df:
                            check_dict = encodeInputs(col)  # encoded col values
                            # st.write(check_dict)
                            # st.write(online_value[0])
                            input_value = online_value[count]  # user input value

                            if input_value in check_dict:
                                # st.write(check_dict[input_value]) #checking input value with the encoded dict
                                online_value[count] = check_dict[input_value]
                            count = count + 1

                        # st.write('encoded input',online_value) # input data has been encoded

                        OnlineInput_df = pd.DataFrame.from_dict([online_value])
                        # Showing the prediction
                        loaded_model = joblib.load(filename)
                        prediction = loaded_model.predict(OnlineInput_df)

                        if st.button('Predict'):
                            if prediction == 1:
                                st.warning('Yes, the customer will terminate the service.')
                            else:
                                st.success('No, the customer will not terminate the service.')

                    if wayToPredict == "Batch":
                        Batch_uploaded_file = st.sidebar.file_uploader("upload the batch file")
                        if Batch_uploaded_file is not None:
                            try:
                                batch_df = pd.read_csv(Batch_uploaded_file)
                                st.write(batch_df.head())
                            except:
                                try:
                                    batch_df = pd.read_excel(Batch_uploaded_file)  # not reading the file
                                    st.write(batch_df.head())
                                except:
                                    st.write("Error: upload the file in csv or excel format")

                            # checking the columns
                            batchCol = set(batch_df.columns)
                            actualCol = set(custom_df.iloc[:, :-1].columns)  # customize columns

                            # st.write(set(batchCol) & set(actualCol))
                            if batchCol == actualCol:
                                batch_objList = []
                                batch_numList = []
                                batch_objNIndexList = []

                                st.write('matched')
                                batch_df1 = batch_df.iloc[:, 1:]  # remove the customer ID col
                                # do type check
                                for col in batch_df1:
                                    if (batch_df.dtypes[
                                        col] == 'O'):  # checking from from batchcol1 and droping in batchcol
                                        batch_objList.append(col)
                                    elif (batch_df.dtypes[col] == 'int64' or batch_df.dtypes[col] == 'float64'):
                                        batch_numList.append(col)

                                # do null check and drop null
                                for col in batch_numList:
                                    if (batch_df[col].isnull().sum() > 0):
                                        batch_df = batch_df.dropna(subset=[col])

                                # dropping null values from the Obj list
                                for col in batch_objList:
                                    if (batch_df[col].str.isspace().sum() > 0):
                                        batch_objNIndexList.append(batch_df[batch_df[col] == ' '].index.values)

                                for row in batch_objNIndexList:
                                    batch_df = batch_df.drop(index=row)  # droping with the index value

                                # batch_df has no null value here
                                batch_df2 = batch_df.iloc[:, 1:]  # ready for encoding, without customerID col


                                # encode
                                def transformEncode(df):
                                    if df[col].nunique() >= 100 and df.dtypes[col] == 'O':
                                        df[col] = pd.to_numeric(df[col], errors='coerce')

                                    return (df)


                                Transform_DF = transformEncode(batch_df2)


                                def encodeInputs(col):
                                    counter = 0
                                    transform_dict = {}
                                    listUnique = Transform_DF[col].unique()

                                    for len_listUnique in range(len(listUnique)):
                                        transform_dict[listUnique[counter]] = counter
                                        counter = counter + 1
                                        # st.write(transform_dict)
                                    return (transform_dict)


                                for col in batch_df2:
                                    if batch_df2.dtypes[col] == 'O':
                                        check_dict = encodeInputs(
                                            col)  # encoded col values in dict form "female":0 "male":1
                                        # st.write(check_dict)
                                        check_dict_list = list(check_dict.keys())  # female and male in list
                                        for lenValue in range(len(check_dict)):
                                            # st.write(check_dict_list[lenValue])
                                            # encoded the batch file
                                            batch_df[col].replace(
                                                {check_dict_list[lenValue]: check_dict[check_dict_list[lenValue]]},
                                                inplace=True)

                                    # scaling
                                    sc = MinMaxScaler()
                                    if batch_df.dtypes[col] == 'int64' or batch_df.dtypes[col] == 'float64':
                                        batch_df[col] = sc.fit_transform(batch_df[[col]])

                                        # st.write(batch_df)

                                # do visualize and predict
                                batch_final_df = batch_df.iloc[:, 1:]
                                # st.write(batch_final_df)


                                if st.button('Predict'):
                                    # Get batch prediction
                                    loaded_model = joblib.load(filename)
                                    prediction = loaded_model.predict(batch_final_df)
                                    prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                                    prediction_df = prediction_df.replace(
                                        {1: 'Yes, the customer will terminate the service.',
                                        0: 'No, the customer will not terminate the service.'})

                                    customerID_df = batch_df.iloc[:, 0:1]
                                    # st.write(customerID_df.head(2))
                                    FinalResult = np.concatenate((customerID_df, prediction_df), axis=1)
                                    FinalResultdf = pd.DataFrame(FinalResult,
                                                             columns=['Customer Identification', 'Prediction'])

                                    st.write(FinalResultdf)


                            else:
                                st.write("batch file's columns does not match with the trained dataset")






































