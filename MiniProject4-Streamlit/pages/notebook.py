# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st
import pydeck as pdk
#%matplotlib inline

st.markdown('# MP4 ')

st.markdown('## 1. Data wrangling and exploration')
# 

st.markdown('### - load and explore the data, clean it, and analyse it by statistics')
# 

# %%
df = pd.read_csv('./data/WA_Fn-UseC_-HR-Employee-Attrition.csv')
st.write('Here is a sample of the data')
st.dataframe(df.sample(10))

st.write('Here is the data description')
st.dataframe(df.describe())

# %%
st.write('Here is the shape of the data')
st.write(df.shape)

# %%
contains_null = True 
for column in df.isnull().sum():
    if column > 0:
        contains_null = True
        break
    else:
        contains_null = False
st.write(f"Does data contain null?: {contains_null}")


# %%
st.write('Here is columns of the data')
st.write(df.columns)

st.write('Keep columns, we think is useful')

columns_to_keep = [ 'Age', 'BusinessTravel', 'Department', 'DistanceFromHome', 'Education', 
                    'EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction', 'MonthlyIncome', 
                    'OverTime', 'WorkLifeBalance', 'YearsAtCompany', 'Attrition', ]
df = df[columns_to_keep]

# %%
columns = df.columns
dtypes = df.dtypes
st.write("Columns that are objects:")
for i in range(len(columns)):
    if(dtypes[i] == 'object'):
        st.write(f"{columns[i]}, :, {dtypes[i]}")

# %%
st.write(df['BusinessTravel'].unique())

# %%
st.write(df['Department'].unique())

# %%
#df['EducationField'].unique()

# %%
#df['Gender'].unique()

# %%
#df['JobRole'].unique()

# %%
#df['MaritalStatus'].unique()

# %%
#df['Over18'].unique()

# %%
st.write(df['OverTime'].unique())

st.markdown('We drop Over18, since the data only has adults')

# %%
#df.drop(['Over18'], axis=1, inplace=True)

st.markdown('We will use label encoding on all the columns, but gender with object to rank them aswell.')

# %%
#df['BusinessTravel'] = df['BusinessTravel'].map({'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2})
#df['Department'] = df['Department'].map({'Sales': 0, 'Research & Development': 1, 'Human Resources': 2})
#df['OverTime'] = df['OverTime'].map({'No': 0, 'Yes': 1})
#df['Attrition'] = df['Attrition'].map({'No': 0, 'Yes': 1})

#df['EducationField'] = df['EducationField'].map({'Life Sciences': 0, 'Medical': 1, 'Marketing': 2, 'Technical Degree': 3, 'Human Resources': 4, 'Other': 5})
#df['JobRole'] = df['JobRole'].map({'Sales Executive': 0, 'Research Scientist': 1, 'Laboratory Technician': 2, 'Manufacturing Director': 3, 'Healthcare Representative': 4, 'Manager': 5, 'Sales Representative': 6, 'Research Director': 7, 'Human Resources': 8})
#df['MaritalStatus'] = df['MaritalStatus'].map({'Single': 0, 'Married': 1, 'Divorced': 2})

st.write('One-hot encoding with the gender column')
encoded_data = pd.get_dummies(df)
df.reset_index()
df = encoded_data
st.dataframe(df.sample(10))


#df.drop(encoded_data.columns, axis=1, inplace=True)
#df.merge(encoded_data)
#df.drop(['Gender'], axis=1, inplace=True)

st.markdown(' ### - select the most relevant features of an employee for machine learning operations on prediction of the attrition')

# %%
st.write(df.columns)

# %%
fig, ax = plt.subplots(figsize=(15,10))
heatmap_data = df.copy()
#plt.subplots()
heatmap = sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.1, ax=ax)
st.write(fig)

st.markdown(' ## Questions for Dora') 
st.markdown('- Is the way we have done encoding correct? We feel like the correlation is not correct, and we are not sure if we should use the label encoding.')

st.markdown(' # 2. Supervised machine learning: classification')
# 

st.markdown(' ### - train, test, and validate two machine learning models for classification and prediction of attrition (e.g Decision Tree and Naïve Bayes)')

st.markdown(' ### imports')

# %%
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
# Uncomment the following line if you don't have Graphviz installed
#%pip install graphviz
#%conda install python-graphviz
import graphviz

st.markdown(' Splitting the data into training and test data')

st.write('We removes the Attrition_Yes and Attrition_No columns, because they hold what we want to predict')
X = df.drop(['Attrition_Yes', 'Attrition_No', 'OverTime_No'], axis=1)
# For y/what we want to predict, we use only the Attrition_Yes column, because Attrition_No holds same data, but inverted
y = df['Attrition_Yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
st.write(X.columns)

st.markdown(' Train Descision Tree model')

# %%
params = {'max_depth': 5}
dt_classifier = DecisionTreeClassifier(**params)
dt_classifier.fit(X_train, y_train)

# %%
gr_data = tree.export_graphviz(dt_classifier, 
                               out_file=None, 
                               feature_names=X.columns, 
                               class_names=True, 
                               filled=True, 
                               rounded=True,
                               proportion = False,
                               special_characters=True)
dtree = graphviz.Source(gr_data)

st.write('Export the tree to a pdf file')
#dtree.render("./docs/AttritionTree") 

st.write('Show the decision tree')
st.graphviz_chart(gr_data)

st.write('Scores the our Descision Tree model against the test data')
dt_classifier.score(X_test, y_test)

st.write('try and predict the test data')
dt_prediction = dt_classifier.predict(X_test)
st.write(dt_prediction)

st.markdown(' Train Naive Bayes model')

# %%
bayes_classifier = GaussianNB()
bayes_classifier.fit(X_train, y_train)

st.write('Scores the our Naïve Bayes model against the test data')
bayes_classifier.score(X_test, y_test)

st.write('try and predict the test data')
bayes_prediction = bayes_classifier.predict(X_test)
st.write(bayes_prediction)

st.markdown('### - apply appropriate methods and measures for assessing the validity of the models and recommend the one with highest accuracy')

st.write('Calculate the accuracy of the models')
dt_accuracy = accuracy_score(y_test, dt_prediction)
bayes_accuracy = accuracy_score(y_test, bayes_prediction)
st.write(f"Decision Tree Accuracy: {dt_accuracy}")
st.write(f"Naïve Bayes Accuracy: {bayes_accuracy}")

st.markdown('With our models, the Decision Tree model has a higher accuracy than the Naive Bayes model. We will in our case recommend the Decision Tree model for predictions.')

st.markdown('## 3. Unsupervised machine learning: clustering')

st.markdown('### - apply at least one clustering algorithm (e.g. K-Means) for segmentation of the employees in groups of similarity')

st.write('Determine k by minimizing the distortion - ')
st.write('the sum of the squared distances between each observation vector and its centroid')
X = X.astype(float)

distortions = []
K = range(2,10)
for k in K:
    KMeans_model = KMeans(n_clusters=k, n_init=10).fit(X)
    KMeans_model.fit(X)
    sum_ = sum(np.min(cdist(X, KMeans_model.cluster_centers_, 'euclidean'), axis=1)) 
    distortions.append( sum_/ X.shape[0]) 
st.write(f"Distortion: {distortions}")

st.write('Plot the distortion to discover the elbow')
plt.clf()
plt.title('Elbow Method for Optimal K')
plt.plot(K, distortions, 'bx-')
plt.xlabel('K')
plt.ylabel('Distortion')
Elbow_fig = plt.gcf()
st.pyplot(Elbow_fig)

st.write('The optimal number of clusters is 4.')
num_clusters = 6

st.write('Create an instance of KMeans classifier')
kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=20)
# init: method of experimemtal finding the initial location of the centroids
# n_init: the algorithm will run n_init times with different cetroids and the best result of those will be taken

st.write('Train the KMeans clustering model')
kmeans.fit(X)

st.write('Predict 4 clusters in X')
y = kmeans.predict(X)
st.write(y)

st.write('See the predicted labels of clusters')
st.write('cluster labels are stored in variable \'kmeans.labels_\'')
st.write(kmeans.labels_)

# same as print(Y)

# %%
st.write(kmeans.cluster_centers_)

st.markdown('We have found optimal number of clusters to be 4, and we can find the center of them,, but since the are higher than 3D we can\'t visualize them.')

st.markdown('### evaluate the quality of the clustering by calculating a silhouette score and recommend the cluster configuration with higher score')
# 

# %%
scores = []
K = range(2,10)
for k in K:
    model = KMeans(n_clusters=k, n_init=10)
    model.fit(X)
    score = metrics.silhouette_score(X, model.labels_, metric='euclidean', sample_size=len(X))
    st.write(f"\nNumber of clusters = {k}")
    st.write(f"Silhouette score = {score}")
    scores.append(score)

st.write('Plot the Silhouette')
plt.clf()
plt.title('Silhouette Score Method for Discovering the Optimal K')
plt.plot(K, scores, 'bx-')
plt.xlabel('K')
plt.ylabel('Silhouette Score')
plt.show()
Silhouette_fig = plt.gcf()   
st.pyplot(Silhouette_fig)

st.markdown('If we are going for the best silhouette score, we would recommend 2 clusters, but if we would want more than 2 clusters, we would recommend 7 clusters.')