import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #for visualization
import plotly.express as px #for visualization
from sklearn.preprocessing import MinMaxScaler


# reading dataset
def fileUpload(message):
    uploadFile = st.sidebar.file_uploader(message)
    if uploadFile is not None:
        try:
            df = pd.read_csv(uploadFile)
            st.write(df.head())
            noOfRows = len(df.axes[0])
            st.write('Number of Rows:', noOfRows)  # print('Number of rows: ', overView_df.shape[0]) #can be done both ways
            noOfCols = len(df.axes[1])
            st.write('Number of Columns:', noOfCols)

        except:
            try:
                df = pd.read_excel(uploadFile)
                st.write(df.head())
                noOfRows = len(df.axes[0])
                st.write('Number of Rows:', noOfRows)  # print('Number of rows: ', overView_df.shape[0]) #can be done both ways
                noOfCols = len(df.axes[1])
                st.write('Number of Columns:', noOfCols)
            except:
                st.write("Uploaded file must be in csv/excel format")
        return df

class Analytics:
    # Analytics of the dataset

    def targetVisualization(self, target_df):
        noOfCols = len(target_df.axes[1])
        targetCol = target_df.columns[noOfCols - 1]  # counting starts from 1 here
        st.write(f'Target Column:`{targetCol}`')
        targetCount = target_df[targetCol].value_counts().to_frame()
        targetCount = targetCount.reset_index()
        targetCount = targetCount.rename(columns={'index': 'Category'})
        Targetfig = px.pie(targetCount, values=targetCol, names='Category',
                           color_discrete_sequence=["green", "red"],
                           title=f'Distribution of {targetCol}')
        return st.plotly_chart(Targetfig)

    def histogram(self, training_df,feature):
        noOfCols = len(training_df.axes[1])
        targetCol = training_df.columns[noOfCols - 1]
        hist_df = training_df.groupby([feature, targetCol]).size().reset_index()
        hist_df = hist_df.rename(columns={0: 'Count'})
        histFig = px.histogram(hist_df, x=feature, y='Count', color=targetCol, marginal='box',
                           title=f'{targetCol} rate frequency to {feature} distribution',
                           color_discrete_sequence=["green", "red"])
        # fig.show()
        return st.plotly_chart(histFig)

class CustomizeCleaning:

    def typeCheck(self,custom_df):
        st.info('Columns and their data types and number of unique values')
        for col in custom_df:
            st.write(f"`{col}` is of {custom_df.dtypes[col]} type and has {custom_df[col].nunique()}")

    def typeChange(self,custom_df):
        typeChange_df = custom_df.iloc[:,1:]
        numList = []
        objList = []
        boolList = []
        #type check
        for col in typeChange_df:
            if typeChange_df.dtypes[col] == 'bool':
                boolList.append(col)
            elif (typeChange_df.dtypes[col] == 'int64' or typeChange_df.dtypes[col] == 'float64'):
                numList.append(col)
            elif typeChange_df.dtypes[col]=='O':
                objList.append(col)
            else:
                st.write(f"{col} of type {typeChange_df.dtypes[col]} is not accepted")
                break

        #if checkPoint1 is not 1:
        #Null Check and drop
        for col in numList:
            if custom_df[col].isnull().sum()>0:
                custom_df.dropna(subset=[col]) #dropping rows
                st.write(f"{col} of numeric data type had {custom_df[col].isnull().sum()} null values : Dropped")

        objNIndexList = []
        for col in objList:
            if custom_df[col].str.isspace().sum() > 0:
                st.write(f'{col} of string data type has {custom_df[col].str.isspace().sum()} null values : Dropped')
                objNIndexList.append(custom_df[custom_df[col] == ' '].index.values)
                for row in objNIndexList:
                    custom_df = custom_df.drop(index=row)  # dropping with the index value

        for col in boolList:
            if custom_df[col].isnull().sum()>0:
                custom_df.dropna(subset=[col]) #dropping rows
                st.write(f"{col} of numeric data type had {custom_df[col].isnull().sum()} null values : Dropped")

            elif custom_df[col].str.isspace().sum() > 0:
                objNIndexList_bool = []
                st.write(f'{col} of string data type has {custom_df[col].str.isspace().sum()} null values : Dropped')
                objNIndexList_bool.append(custom_df[custom_df[col] == ' '].index.values)
                for row in objNIndexList_bool:
                    custom_df = custom_df.drop(index=row)  # dropping with the index value

        st.info("Customize data types")
        for col in numList:
            st.write(f"{col} of {custom_df.dtypes[col]} type")
            changeToString = st.checkbox(f"Change {col} to string")
            if changeToString:
                custom_df[col] = custom_df[col].astype(str)

        for col in objList:
            st.write(f"{col} of {custom_df.dtypes[col]} type")
            changeToFloat = st.checkbox(f"Change {col} to float")
            if changeToFloat:
                try:
                    custom_df[col] = custom_df[col].astype(float)
                except:
                    st.write('You can encode it in the next step')
            changeToInt = st.checkbox(f"Change {col} to integer")
            if changeToInt:
                try:
                    custom_df[col] = custom_df[col].astype(int)
                except:
                    st.write('You can encode it in the next step')

        return(custom_df)

    def encode(self,custom_df1):
        encode_df = custom_df1.iloc[:,1:]
        EnumList = []
        EobjList = []
        EboolList = []
        # type check
        for col in encode_df:
            if encode_df.dtypes[col] == 'bool':
                EboolList.append(col)
            elif (encode_df.dtypes[col] == 'int64' or encode_df.dtypes[col] == 'float64'):
                EnumList.append(col)
            elif encode_df.dtypes[col] == 'O':
                EobjList.append(col)

        #encodeCatagorial
        for col in EobjList:
            counter = 0
            uniqueList = custom_df1[col].unique()
            unique_df = pd.DataFrame(uniqueList)
            unique_df.columns = [col]

            df = unique_df
            lenOfUniquness = len(df)
            for i in range(lenOfUniquness):
                df_value = (df.iloc[i][col])
                # print(df_value)
                custom_df1[col].replace({df_value: counter}, inplace=True)
                counter = counter + 1

        # Scaling
        sc = MinMaxScaler()
        for col in EnumList:
            custom_df1[col] = sc.fit_transform(custom_df1[[col]])

        st.info("Encoded Dataset")
        st.write(custom_df1)  # encode check
        st.write('shape of encode_df', custom_df1.shape)

        return custom_df1

    def outlierCheck(self,encoded_df):
        outlierCheck_df = encoded_df.iloc[:,1:]
        #st.write(outlierCheck_df)
        for col in outlierCheck_df:
            if len(pd.unique(outlierCheck_df[col])) > 2:
                meanValue = outlierCheck_df[col].mean()
                stdValue = outlierCheck_df[col].std()

                # 2nd standard deviation
                upperLimit = meanValue + 2 * stdValue
                # st.write(f'Upper limit : {upperLimit}')
                lowerLimit = meanValue - 2 * stdValue
                # st.write(f'Lower limit : {lowerLimit}')

                outliersDetected_df = outlierCheck_df[(outlierCheck_df[col] >= upperLimit) | (outlierCheck_df[col] <= lowerLimit)]
                #st.write('outliersDetected_df', outliersDetected_df)
                if not outliersDetected_df.empty:
                    # st.write(f'{fcol} has outliers : ')
                    # storing the index number of the outliers
                    st.write(f'Upper Limit of {col} : {upperLimit}')
                    st.write(f'Lower Limit of {col} : {lowerLimit}')
                    st.write(f'{col} has {len(outliersDetected_df[col])} outliers')
                    st.write(outliersDetected_df[col])

                    dropOutliers = st.checkbox(f'Do you want to drop the outliers of {col}')
                    if dropOutliers:
                        indexNum = []
                        indexNum = outliersDetected_df.index.values.tolist()
                        # st.write('index list',indexNum)
                        st.write('to drop shape', outliersDetected_df.shape)

                        for row in indexNum:
                            # st.write(row)
                            encoded_df = encoded_df.drop(index=row)
                            # st.write(f_df.shape)

                        st.write('shape of DataFrame : ', encoded_df.shape)
                        return encoded_df












































































