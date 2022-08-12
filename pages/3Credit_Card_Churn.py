import streamlit as st

import pandas as pd
import numpy as np

data = pd.read_csv("BankChurners.csv")
data.drop(["CLIENTNUM"], axis=1, inplace=True)

st.title('Credit Cards: Cause and Prediction')
st.write('Determine Cause and Predict possible Customer Attrition')

st.subheader('Context')
st.write('Data Shows a steep decline in the number of credit card users. By exploring the data on existing and past customers we can identify customer reseans for leaving and build a model to predict who will')

with st.sidebar:
    st.header('Variable Dictionary')
    st.write('The dataset has the following information:')
    st.markdown('* CLIENTNUM: Client number. Unique identifier for the customer holding the account \n* Attrition_Flag: Internal event (customer activity) variable: if the account is closed then "Attrited Customer" else "Existing Customer" \n* Customer_Age: Age in Years\n* Gender: Gender of the account holder\n* Dependent_count: Number of dependents \n* Education_Level: Educational Qualification of the account holder: Graduate, High School, Unknown, Uneducated, College(refers to college student), Post-Graduate, Doctorate\n* Marital_Status: Marital Status of the account holder\n* Income_Category: Annual Income Category of the account holder\n* Card_Category: Type of Card\n* Months_on_book: Period of relationship with the bank (in months)\n* Total_Relationship_Count: Total no. of products held by the customer\n* Months_Inactive_12_mon: No. of months inactive in the last 12 months\n* Contacts_Count_12_mon: No. of Contacts in the last 12 months\n* Credit_Limit: Credit Limit on the Credit Card\n* Total_Revolving_Bal: Total Revolving Balance on the Credit Card\n* Avg_Open_To_Buy: Open to Buy Credit Line (Average of last 12 months)\n* Total_Amt_Chng_Q4_Q1: Change in Transaction Amount (Q4 over Q1)\n* Total_Trans_Amt: Total Transaction Amount (Last 12 months)\n* Total_Trans_Ct: Total Transaction Count (Last 12 months)\n* Total_Ct_Chng_Q4_Q1: Change in Transaction Count (Q4 over Q1)\n* Avg_Utilization_Ratio: Average Card Utilization Ratio\n#### What is a Revolving Balance?\n* If we don\'t pay the balance of the revolving credit account in full every month, the unpaid portion carries over to the next month. That\'s called a revolving balance.\n#### What is the Average Open to buy?\n* Open to Buy means the amount left on your credit card to use. Now, this column represents the average of this value for the last 12 months.\n#### What is the Average utilization Ratio?\n* The Avg_Utilization_Ratio represents how much of the available credit the customer spent. This is useful for calculating credit scores.\n#### Relation b/w Avg_Open_To_Buy, Credit_Limit and Avg_Utilization_Ratio:\n* ( Avg_Open_To_Buy / Credit_Limit ) + Avg_Utilization_Ratio = 1')

import seaborn as sns
import matplotlib.pyplot as plt

tab1, tab2 = st.tabs(["Analysis", "Modeling"])

with tab1:
    st.header('Univariate Analysis')
    variable = st.selectbox('Select Variable',data.columns)

    if data[variable].dtypes == 'object':
        fig = plt.figure()
        temp = data[variable].value_counts().sort_index()
        sns.barplot(x=temp.index,y=temp.values)
        st.pyplot(fig) 

    else:
        col1, col2= st.columns([1,1])
        with col1:
            fig = plt.figure()
            sns.boxplot(data[variable])
            st.pyplot(fig)   
        with col2:
            fig = plt.figure()
            sns.distplot(data[variable])
            st.pyplot(fig)

    st.subheader('Univariate Analysis: Summary')
    st.write('**Credit limit, Average open to buy and Average utilization ratio are right-skewed**.\n1. **Open to buy** means how much credit a customer is left with\n* Low values of Open to buy could represent either\n* Customers have low credit limits\n* Customers are spending a lot so they are left less open to buy\n2. **Average utilization ratio** = (1 - (open to buy/credit limit))\n* Low values of the Average utilization ratio represents\n* (Open to buy/credit limit) is nearly equal to 1 -> Open to buy is nearly equal to the credit limit -> customers are spending less using their credit cards\n3. **Credit limit** is also right-skewed which represents - most of the customers have low credit limits\n Based on the three variables, we can conclude that the majority of customers have low credit limits and are not utilizing their credit cards frequently. Now this statement justifies the right skewness for all the three variables.')

    st.header('Bivariant Analysis')

    st.subheader('Correlation Analysis')
    fig = plt.figure(figsize=(20,10))
    sns.heatmap(data.corr(), annot=True)
    st.pyplot(fig)

    st.write('**Observations:**\n* Attrition_Flag shows a bit of a negative correlation with **Total_Trans_Ct (total transactions)** and **Total_Trans_Amt (total transaction amount)**.\n* There\'s a strong positive correlation between **Months_on_book** and **Customer_Age**, **Total_Revolving_Bal** and **Avg_Utilization_Ratio**, **Total_Trans_Amt** and **Total_Trans_Ct**.\n* There\'s a negative correlation of **Total_Relationship_count** with **Total_Trans_Amt** and **Total_Trans_Ct**, **Avg_Utilization_Ratio** with **Avg_Open_To_Buy** and **Credit_Limit**.')

    st.subheader('Categorical Attrition Analysis')
    st.write('Percent of each Category that has Attritioned')
    variable = st.selectbox('Select Variable',data.select_dtypes(include='object').columns)

    temp = pd.crosstab(index=data[variable],columns=data['Attrition_Flag'],normalize="index")
    st.write(temp)
    fig = plt.figure()
    sns.barplot(x = temp.index,y = temp['Attrited Customer'].values)
    st.pyplot(fig)

    st.write('**Observations:**')

with tab2:
    st.header('Data Preparation')
    data["Attrition_Flag"].replace("Existing Customer", 0, inplace=True)
    data["Attrition_Flag"].replace("Attrited Customer", 1, inplace=True)
    data["Income_Category"].replace("abc", np.nan, inplace=True)
    
    with st.expander('Variable Summary Statistics'):
        st.write(data.describe())
        for i in data.describe(include=["object"]).columns:
            st.write("Unique values in", i, "are :")
            st.write(data[i].value_counts())
            st.write("*" * 50)
    
    col1, col2= st.columns([1,1])
    with col1:
        st.subheader('Percent missing in each Column')
        st.dataframe(round(data.isnull().sum() / data.isnull().count() * 100, 2))
    with col2:
        st.subheader('Percent that are outliers in each Column')
        IQR = data.quantile(0.75) - data.quantile(0.25)
        lower = (data.quantile(0.25) - 1.5 * IQR)
        upper = (data.quantile(0.75) + 1.5 * IQR)
        st.dataframe(((data.select_dtypes(include=["float64", "int64"]) < lower)| (data.select_dtypes(include=["float64", "int64"]) > upper)).sum() / len(data) * 100)
    
    from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score,confusion_matrix,roc_auc_score,plot_confusion_matrix,classification_report,precision_recall_curve

    from sklearn import metrics

    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

    # To be used for data scaling and one hot encoding
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

    # To impute missing values
    from sklearn.impute import SimpleImputer

    # To help with model building
    from sklearn.linear_model import LogisticRegression

    # To build classification models 
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression


    # For tuning the model
    from sklearn.model_selection import GridSearchCV

    X = data.drop(["Attrition_Flag"], axis=1)

    reqd_col_for_impute = ["Education_Level", "Marital_Status", "Income_Category"]
    imputer = SimpleImputer(strategy="most_frequent")
    X[reqd_col_for_impute] = imputer.fit_transform(X[reqd_col_for_impute])
    X = pd.get_dummies(X, drop_first=True)

    y = data["Attrition_Flag"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)

    st.header('Modeling')
    st.write('### **Model evaluation criterion**\n**The model can make two types of wrong predictions:**\n1. Predicting a customer will attrite and the customer doesn\'t attrite.\n2. Predicting a customer will not attrite and the customer attrites.\n**Which case is more important?**\n* Predicting that customer will not attrite but he attrites i.e. losing on a valuable customer or asset.\n**How to reduce this loss i.e the need to reduce False Negatives?**\n* **Bank would want `Recall` to be maximized**, greater the Recall higher the chances of minimizing false negatives. Hence, the focus should be on increasing Recall or minimizing the false negatives or in other words identifying the true positives (i.e. Class 1) so that the bank can retain their valuable customers by identifying the customers who are at risk of attrition.')

    def metrics_score(actual, predicted):
        st.dataframe(classification_report(actual, predicted,output_dict=True))
        cm = confusion_matrix(actual, predicted)
        fig = plt.figure(figsize=(8,5))
        sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels=['Not Attrite', 'Attrite'], yticklabels=['Not Attrite', 'Attrite'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)

    st.subheader('Logistic Regression')

    lg = LogisticRegression()
    lg.fit(X_train,y_train)
    st.write('train')
    y_pred_train = lg.predict(X_train)
    metrics_score(y_train, y_pred_train)

    st.write('test')
    y_pred_test = lg.predict(X_test)
    metrics_score(y_test, y_pred_test)
    
    st.write('Regression Coefficients')
    cols=X_train.columns
    coef_lg=lg.coef_
    st.dataframe(pd.DataFrame(coef_lg,columns=cols).T.sort_values(by=0,ascending=False))
    st.markdown('**Observations:**\n**Features which positively affect on the attrition rate are:**\n- Contacts_Count_12_mon\n- Months_Inactive_12_mon\n- Dependent_count\n- Customer_Age\n- Income_Category_Less than $40K	\n- Marital_Status_Single\n- Education_Level_Graduate	\n- Education_Level_Post-Graduate\n- Education_Level_Doctorate\n- Avg_Utilization_Ratio\n**Features which negatively affect on the attrition rate are:**\n- Total_Relationship_Count	\n- Total_Trans_Ct\n- Months_on_book\n- Total_Ct_Chng_Q4_Q1\n- Marital_Status_Married\n- Income_Category_ 60K-80K\n- Total_Amt_Chng_Q4_Q1')

    st.write('Interperate Coefficients in real world terms')
    odds = np.exp(lg.coef_[0])
    st.dataframe(pd.DataFrame(odds, X_train.columns, columns=['odds']).sort_values(by='odds', ascending=False))
    st.markdown('**Observations**\n- The odds of a customers contacting with the bank more to attrite are **1.3 times** the odds of one who is not, probably due to the fact that the bank is not able to resolve the problems faced by customers leading to attrition.\n- The odds of a customer being inactive to attrite are **1.2 times** the odds of a customer who is actively in touch with bank.\n- The odds of a customer with more dependent attriting are **1.2 times** the odds of a customer with less or no dependent.')

    st.write('Precision-Recall Curve')
    st.write('Precision-Recall curves summarize the trade-off between the true positive rate and the positive predictive value for a predictive model using different probability thresholds.')
    y_scores_lg=lg.predict_proba(X_train)
    precisions_lg, recalls_lg, thresholds_lg = precision_recall_curve(y_train, y_scores_lg[:,1])
    fig = plt.figure()
    plt.plot(thresholds_lg, precisions_lg[:-1], 'b--', label='precision')
    plt.plot(thresholds_lg, recalls_lg[:-1], 'g--', label = 'recall')
    plt.xlabel('Threshold')
    plt.legend(loc='upper left')
    plt.ylim([0,1])
    st.pyplot(fig)


    hyperparam_grid = {"class_weight": [{0:1,1:4},{0:3,1:7},{0:1,1:3}]
                    ,"penalty": ["l1", "l2"]
                    ,"C": np.arange(5, 7, 1)
                    ,"fit_intercept": [True, False]  }
    # define model
    lg3 = LogisticRegression()
    # define evaluation procedure
    grid = GridSearchCV(lg3,hyperparam_grid,scoring="precision", cv=100, n_jobs=-1, refit=True)
    grid.fit(X,y)
    st.write(f'Best score: {grid.best_score_} with param: {grid.best_params_}')
    
    lg3 = LogisticRegression(C=6.0,fit_intercept=True, penalty='l2',class_weight={0: 3, 1: 7} )
    lg3.fit(X_train,y_train)
    st.write('train')
    y_pred_train = lg3.predict(X_train)
    metrics_score(y_train, y_pred_train)
    st.write('test')
    y_pred_test = lg3.predict(X_test)
    metrics_score(y_test, y_pred_test)

    st.subheader('Decision Tree')


st.header('Ending Summary')
st.write('* We have been able to build a predictive model:\na) that the bank can deploy to identify customers who are at risk of attrition.\nb) that the bank can use to find the key causes that drive attrition. \nc) based on which the bank can take appropriate actions to build better retention policies for customers.\n* Factors that drive attrition - Total_Trans_Ct, Total_Revolving_Bal, Total_Trans_Amt, Total_Relationship_Count\n* Total_Trans_Ct: Less number of transactions in a year leads to attrition of a customer - to increase the usage of cards the bank can provide offers like cashback, special discounts on the purchase of something, etc so that customers feel motivated to use their cards.\n* Total_Revolving_Bal: Customers with less total revolving balance are the ones who attrited, such customers must have cleared their dues and opted out of the credit card service. After the customer has cleared the dues bank can ask for feedback on their experience and get to the cause of attrition.\n* Total_Trans_Amt: Less number of transactions can lead to less transaction amount and eventually leads to customer attrition - Bank can provide offers on the purchase of costlier items which in turn will benefit the customers and bank both.\n* Total_Relationship_Count: Attrition is highest among the customers who are using 1 or 2 products offered by the bank - together they constitute ~55% of the attrition - Bank should investigate here to find the problems customers are facing with these products. Customer support, or more transparency can help in retaining customers.\n* Female customers should be the target customers for any kind of marketing campaign as they are the ones who utilize their credits, make more and higher amount transactions. But their credit limit is less, so increasing the credit limit for such customers can profit the bank.\n* Months_Inactive: As inactivity increases the attrition also increases. 2-4 months of inactivity are the biggest contributors of attrition - Bank can send automated messages to engage customers, these messages can be about their monthly activity, new offers or services, etc.\n* Highest attrition is among the customers who interacted/reached out the most with/to the bank, this indicates that the bank is not able to resolve the problems faced by customers leading to attrition - a feedback collection system can be set up to check if the customers are satisfied with the resolution provided, if not, the bank should act upon it accordingly.')