import streamlit as st

def generalDescription():
    st.title("`Documentation`")
    st.markdown(""" 
    This application can detect the Churn Prediction on any company's dataset.User can view analytics & 
    customize the training dataset before doing the prediction.
    """)

    st.info("Features on training dataset:")
    st.markdown(""" 1)Can view the analytics on the overall dataset \n 
    2)Churn(target column) rate analytics \n
    3)Visualize the histogram of a selected attribute against Churn(target) \n
    4)Can drop any attribute and customize the training dataset \n
    5)View list of unique value numbers of each attribute \n
    6)Null check will do done automatically,if null found it will be dropped and informed about the dropping \n
    7)Attribute type check will be done automatically \n
    8)Training dataset will be encoded automatically \n
    9)List of outliers will be shown, user can choose to drop or not \n
    10)Customize the test-train ration \n
    11)Choose a desired ML or Deep Learning model.Models used here are: \n
      (i) k-nearest neighbors(KNN)
      (ii)Support-vector machine(SVM)
      (iii)Decision Tree(DT)
      (iv)Random Forest Classifier(RFC)
      (v)Artificial Neural Network(ANN)
    12)Tune the parameter's values of the choose model and view accuracy of the model
    """)

    st.info("For Prediction there are 2 categories available")
    st.markdown(""" 
     `Online Prediction:` here the prediction can be done for a single user by giving all the attributes values. \n
     `Batch Prediction:` here user can upload a new dataset following the similar attribute structure of the 
     training dataset to view the prediction.
    """)

    st.info("Constrains")
    st.markdown("""
    1) Uploaded dataset should be within the size of 200MB \n
    2) First column of the uploaded dataset must be customer identification column only \n
    3) Last column of the training dataset must be the target column \n
    4) No more than 100 unique value in any string type categories attribute \n
    5) If null values found the batch prediction dataset, it will be dropped automatically. \n
    """)

    st.markdown("<h3></h3>", unsafe_allow_html=True)

