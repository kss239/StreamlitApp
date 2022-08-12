import streamlit as st

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


pima = pd.read_csv("diabetes.csv")

st.title('Determining Diabetes in Pima Native Americans')
st.header('Context')
st.write('Diabetes is one of the most frequent diseases worldwide and the number of diabetic patients are growing over the years. The main cause of diabetes remains unknown, yet scientists believe that both genetic factors and environmental lifestyle play a major role in diabetes. A few years ago research was done on a tribe in America which is called the Pima tribe. In this tribe, it was found that the women are prone to diabetes very early. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients were females at least 21 years old of Pima Indian heritage.')

with st.sidebar:
    st.header('Variable Dictionary')
    st.write('The dataset has the following information:')

    st.markdown(
    '* Pregnancies: Number of times pregnant\n* Glucose: Plasma glucose concentration over 2 hours in an oral glucose tolerance test\n* BloodPressure: Diastolic blood pressure (mm Hg)\n* SkinThickness: Triceps skin fold thickness (mm)\n* Insulin: 2-Hour serum insulin (mu U/ml)\n* BMI: Body mass index (weight in kg/(height in m)^2)\n* DiabetesPedigreeFunction: A function that scores the likelihood of diabetes based on family history.\n* Age: Age in years\n* Outcome: Class variable (0: a person is not diabetic or 1: a person is diabetic)'
    )

st.header('Data Overview')

col1, col2= st.columns([1,1])
with col1:
    fig = plt.figure()
    sns.boxplot(data=pima.drop(['Outcome'],axis = 1), palette="Set2")
    plt.xticks(rotation = 85)
    st.pyplot(fig)
    
with col2:
    corr_matrix = pima.iloc[:,0:9].corr()
    fig = plt.figure()
    sns.heatmap(corr_matrix.iloc[:,:], annot = True)
    plt.xticks(rotation = 85)
    st.pyplot(fig)

with st.expander('Variable Summary Statistics'):
    st.dataframe(pima.iloc[:,0:8].describe())

col1, col2, col3= st.columns([1,2,2])
with col1:
    option = st.selectbox('Select Variable Distribution',pima.drop(['Outcome'],axis = 1).columns)
    agree = st.checkbox('Seperate by Diabetic Outcome')

with col2:
    if agree:
        fig = sns.displot(pima, x=option, hue="Outcome", kind="kde", fill=True)
        st.pyplot(fig)
    else:
        fig = sns.displot(pima, x=option, kind="kde", fill=True)
        st.pyplot(fig)
with col3:
    if agree:
        fig = plt.figure()
        sns.boxplot(data = pima, x=pima['Outcome'],y=pima[option], palette="Set2")
        plt.xticks(rotation = 85)
        st.pyplot(fig)
    else:
        fig = plt.figure()
        sns.boxplot(data = pima, x=pima[option], palette="Set2")
        plt.xticks(rotation = 85)
        st.pyplot(fig)


st.header('Predict Diabetes')
st.subheader('Build model')
st.write('With the previous diabetes data, below is a logistic regression model that can predict diabetes status.')

col1, col2= st.columns([1,1])
with col1:
    value = st.slider(
        'Model Variable Correlation threshold',
        0.0, float(corr_matrix['Outcome'].nlargest(2).iloc[1]), 0.2)
    st.write('select what the minimum correlation of Variables to Diabetes to be included in this model')

with col2:
    options = []
    for i in range(len(corr_matrix['Outcome'])):
        if corr_matrix['Outcome'][i] >= value:
            options.append(corr_matrix['Outcome'].index[i])
    options.remove('Outcome')
    st.write('Variables include:')
    st.write(options)

st.subheader('Use model to make predict')
st.write('Input values to predict probability of Diabetes')

x_train, x_test, y_train, y_test = train_test_split(pima[options], pima['Outcome'], test_size=0.2, random_state=0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

model = LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr',random_state=0)
model.fit(x_train, y_train)

x_test = scaler.transform(x_test)
y_pred = model.predict(x_test)

accuracy = round((model.score(x_train, y_train)+model.score(x_test, y_test))/2,3)


col1, col2= st.columns([1,3])
with col1:
    input_values = []
    for x in options:
        input_values.append(st.number_input(x, min_value=float(pima[x].min()), max_value=float(pima[x].max()),value=float(pima[x].median())))

    new_data = np.array([input_values])    
    new_data_scaled = scaler.transform(new_data)
    prob=model.predict_proba(new_data_scaled)


with col2:
    
    fig = plt.figure()
    sns.barplot(x = ["Not Diabetic","Diabetic"],y = [prob[0][0],prob[0][1]], palette="Set2").set(title=f'Predicted Probability of being Diabetic, acc({accuracy})')
    st.pyplot(fig)

