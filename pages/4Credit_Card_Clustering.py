import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pylab as plt
import seaborn as sns

#to scale the data using z-score 
from sklearn.preprocessing import StandardScaler

#importing clustering algorithms
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from sklearn_extra.cluster import KMedoids

from sklearn.metrics import silhouette_score

st.title('Credit Card Customer Clustering')


st.write('AllLife Bank wants to focus on its credit card customer base in the next financial year. They have been advised by their marketing research team, that the penetration in the market can be improved. Based on this input, the Marketing team proposes to run personalized campaigns to target new customers as well as upsell to existing customers. Another insight from the market research was that the customers perceive the support services of the bank poorly. Based on this, the Operations team wants to upgrade the service delivery model, to ensure that customers queries are resolved faster. The Head of Marketing and Head of Delivery both decide to reach out to the Data Science team for help.')

with st.sidebar:
    st.header('Data Dictionary')
    st.write('Data is of various customers of a bank with their credit limit, the total number of credit cards the customer has, and different channels through which customer has contacted the bank for any queries, different channels include visiting the bank, online and through a call centre.')
    st.markdown('* Sl_no -Customer Serial Number (Absent from Analysis)\n* Customer Key -Customer identification (Absent from Analysis)\n* Avg_Credit_Limit -Average credit limit (currency is not specified, you can make an assumption around this)\n* Total_Credit_Cards -Total number of credit cards \n* Total_visits_bank -Total bank visits\n* Total_visits_online -Total online visits\n* Total_calls_made -Total calls made')

data = pd.read_excel('Credit+Card+Customer+Data.xlsx')
data.drop(columns = ['Sl_No', 'Customer Key'], inplace = True)
data.drop_duplicates()

st.header('Data Analysis')

st.subheader('Variable analysis')
with st.expander('Variable Summary Statistics'):
    st.dataframe(data.describe())
    st.markdown('* The average credit limit has a high range as it has a minimum value of 3K and a maximum value of 200K.\n*The mean of the average credit limit is approx 34.5K with a large standard deviation of 37.4K.\n* The average number of cards per customer is approx 5 (rounding off to nearest integer).\n* On average, a customer has 2 bank visits, 3 online visits, and made 4 calls.')

option = st.selectbox(
     'Select Variable',
     data.columns)
st.write('Skew :',round(data[option].skew(),2),'\nExcess Kurtosis :',round(data[option].kurt(),2))
    
col1,col2= st.columns(2)
with col1:
    fig = plt.figure()
    data[option].hist(bins=10, grid=False,color='purple')
    plt.xlabel(option)
    plt.ylabel('count')
    st.pyplot(fig)

with col2:
    fig = plt.figure()
    plt.title('')
    sns.boxplot(x=data[option],color='pink')
    st.pyplot(fig)

col1,col2= st.columns([1,2])
with col1:
    st.write('')
    sub_option1 = st.selectbox(
     'Vary Against',
     data.columns)
    sub_options2 = st.selectbox(
     'Color Against',
     data.columns)
with col2:
    fig = sns.jointplot(x=data[option], y=data[sub_option1], hue=data[sub_options2], kind='scatter')
    st.pyplot(fig)

st.subheader('Variable Correlation Heatmap')# Add Pop out Correlation Scatter plot

fig = plt.figure()
sns.heatmap(data.corr(), annot=True, fmt='0.2f')
st.pyplot(fig)

st.markdown('* Avg_Credit_Limit is positively correlated with Total_Credit_Cards and Total_visits_online which can makes sense.\n* Avg_Credit_Limit is negatively correlated with Total_calls_made and Total_visits_bank.\n* Total_visits_bank, Total_visits_online, Total_calls_made are negatively correlated which implies that majority of customers use only one of these channels to contact the bank.')



st.header('Clustering')

scaler=StandardScaler()
data_scaled=pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
data_scaled_copy = data_scaled.copy(deep=True)
st.subheader('Select Clustering Method')


sc = {}
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k).fit(data_scaled)
    labels = kmeans.predict(data_scaled)
    sc[k] = silhouette_score(data_scaled, labels)

n = 0
s_score = 0
for key, value in sc.items():
    if value > s_score:
        s_score = value
        n = key

#Kmean
kmeans = KMeans(n_clusters=n, max_iter= 1000, random_state=1)
kmeans.fit(data_scaled)

data_scaled_copy['Labels'] = kmeans.predict(data_scaled)
data['Labels'] = kmeans.predict(data_scaled)

#Kmedo
kmedo = KMedoids(n_clusters = n, random_state=1)
kmedo.fit(data_scaled)

data_scaled = np.ascontiguousarray(data_scaled)
data_scaled_copy['kmedoLabels'] = kmedo.predict(data_scaled)
data['kmedoLabels'] = kmedo.predict(data_scaled)

#Gmm
gmm = GaussianMixture(n_components = n)
gmm.fit(data_scaled)

data_scaled_copy['GmmLabels'] = gmm.predict(data_scaled)
data['GmmLabels'] = gmm.predict(data_scaled)


def tab_scatterplot(color):
    col1,col2= st.columns([1,2])
    with col1:# come back to this
        x_axis = st.selectbox(
        'X axis',
        data.columns[:-3],key=f'x{color}')
        y_axis = st.selectbox(
        'Y axis',
        data.columns[:-3],key=f'y{color}')

    with col2:
        if x_axis == y_axis:
            fig = sns.displot(x=data[x_axis], col=data[color],kind="kde")
            st.pyplot(fig)
        else:
            fig = sns.displot(x=data[x_axis], y=data[y_axis], col=data[color],kind="kde")
            st.pyplot(fig)

tab1, tab2, tab3 = st.tabs(['K-Means','K-Medoids','Gaussian Mixture'])

with tab1:
    tab_scatterplot('Labels')
with tab2:
    tab_scatterplot('kmedoLabels')
with tab3:
    tab_scatterplot('GmmLabels')
