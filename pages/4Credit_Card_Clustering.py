from os import scandir
import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pylab as plt
import seaborn as sns

#to scale the data using z-score 
from sklearn.preprocessing import StandardScaler

#importing clustering algorithms
from sklearn.cluster import KMeans

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
kmeans = KMeans(n_clusters=n, max_iter= 1000, random_state=1)#n = 3
kmeans.fit(data)
centroids = kmeans.cluster_centers_

data['Labels'] = kmeans.predict(data)

#Mapping to only 3
colors = ['#DF2020', '#81DF20', '#2095DF']
data['c'] = data['Labels'].map({0:colors[0], 1:colors[1], 2:colors[2]})

from matplotlib.lines import Line2D

def tab_scatterplot():
    col1,col2= st.columns([1,2])
    with col1:# come back to this
        x_axis = st.selectbox(
        'X axis',
        data.columns[:-3],key='x'+'Labels')
        y_axis = st.selectbox(
        'Y axis',
        data.columns[:-3],key='y'+'Labels')

    with col2:
        if x_axis == y_axis:
            st.header('Choose different X Axis & Y Axis')
        else:
            fig, ax = plt.subplots(1, figsize=(8,8))
            # plot data
            plt.scatter(data[x_axis], data[y_axis], c=data.c, alpha = 0.6, s=10)
            # plot centroids
            cen_x = [i[data.columns.get_loc(x_axis)] for i in centroids] 
            cen_y = [i[data.columns.get_loc(y_axis)] for i in centroids]
            data['cen_x'] = data['Labels'].map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2]})
            data['cen_y'] = data['Labels'].map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2]})
            plt.scatter(cen_x, cen_y, marker='^', c=colors, s=70)
            # plot lines
            for idx, val in data.iterrows():
                x = [val[x_axis], val.cen_x,]
                y = [val[y_axis], val.cen_y]
                plt.plot(x, y, c=val.c, alpha=0.2)
            # legend
            legend_elements = [Line2D([0], [0], marker='o', color='w', label='Cluster {}'.format(i+1), 
                            markerfacecolor=mcolor, markersize=5) for i, mcolor in enumerate(colors)]
            legend_elements.extend([Line2D([0], [0], marker='^', color='w', label='Centroid - C{}'.format(i+1), 
                        markerfacecolor=mcolor, markersize=10) for i, mcolor in enumerate(colors)])
            
            plt.legend(handles=legend_elements, loc='upper right', ncol=2)

            plt.xlabel(x_axis)
            plt.ylabel(y_axis)

            st.pyplot(fig)

tab_scatterplot()
