#importing required Modules
from Modules import *
path='models/'

# dataset for model1
df1 = pd.read_csv("Datasets/student_marks.csv")
# independent & dependent data
x = df1.iloc[:,:-1].values
y = df1.iloc[:, -1].values
# creating object of linear regression class
model = LinearRegression()
# fitting data into model
model.fit(x,y)
#Dump fitted model into pickle file
pickle.dump(model, open(path + 'student_marks_model.pkl', 'wb'))
print("Model 1 saved succesfully")




# dataset for model2
df2 = pd.read_csv("Datasets/iris.csv")
# dropping unneccesary data column
df2.drop('Id', axis=1, inplace=True)
# 25 and 75 percentile
q1 = df2['SepalWidthCm'].quantile(0.25)
q3 = df2['SepalWidthCm'].quantile(0.75)
# inter quartile range
iqr = q3 - q1
# removing outliers
df2 = df2[(df2['SepalWidthCm'] >= q1 - 1.5 * iqr) & (df2['SepalWidthCm'] <= q3 + 1.5 * iqr)]
# train dataset
X = df2.iloc[:, 0:4]
# scaling dataset in proper scale
scale = StandardScaler()
norm_df = scale.fit_transform(X)
# finding clusters number
cluster_range = range(1, 20)
# ( Within-Cluster Sum of Square ).
WCSS = []
for n_cluster in cluster_range:
    clusters = KMeans(n_cluster, n_init=10)
    clusters.fit(X)
    labels = clusters.labels_
    center = clusters.cluster_centers_
    WCSS.append(clusters.inertia_)
# model trainig using k-means clustering
species_model = KMeans(n_clusters=3, max_iter=300)
species_model.fit(X)
#Dump fitted model into pickle file
pickle.dump(species_model, open(path + 'iris_model.pkl', 'wb'))
print("Model 2 saved Succesfully")





# dataset for model 3
#dataset = pd.read_csv("Datasets/breast cancer dataset kaggle.csv")
dataset = sklearn.datasets.load_breast_cancer()
df3=pd.DataFrame(dataset.data, columns=dataset.feature_names)
#data Standardaization
dataset.data.mean()
dataset.data.std()
scaler=StandardScaler()
scaler.fit(dataset.data)
std_data=scaler.transform(dataset.data)
std_data.mean()
std_data.std()
# independent & dependent dataset
x=std_data
y=dataset.target
# splitting data into train & test data
X_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify = y,random_state=2)
# logistic regression object
model = LogisticRegression()
# fitting data into model
model.fit(X_train, y_train)
#Dump fitted model into pickle file
pickle.dump(model, open(path + 'cancer_prediction_model.pkl', 'wb'))
print("Model 3 saved Succesfully")




#dataset for model 4
raw_mail_data = pd.read_csv("Datasets/spamham.csv")
#replace the null values with a null string
df4=raw_mail_data.where((pd.notnull(raw_mail_data)), '')
#Mail labeling
df4.loc[df4['Category'] == 'spam', 'Category',]=0
df4.loc[df4['Category'] == 'ham', 'Category',]=1
#data sepratation
X=df4['Message']
Y=df4['Category']
# split the data as train data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=3)
# transform the text data to feature vectors that can be used as input to the SVM model using TfidfVectorizer
# convert the text to lower case letters
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase='True')
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
#convert Y_train and Y_test values as integers
Y_train =Y_train.astype('int')
Y_test = Y_test.astype('int')
#dump vectorizer
pickle.dump(feature_extraction, open('feature_extraction_vect.pkl', 'wb'))
# training the support vector machine model with training data
model = LinearSVC()
model.fit(X_train_features, Y_train)
#Dump fitted model into pickle file
pickle.dump(model, open(path + 'spam_mail_prediction_model.pkl', 'wb'))
print("Model 4 saved Succesfully")





#dataset for model 5
df5 = pd.read_csv("Datasets/titanic_survival.csv")
#labeling of data 
income_set=set(df5['Sex'])
df5['Sex']=df5['Sex'].map({'female':0,'male':1}).astype(int)
# independent & dependent data
x=df5.drop(['Survived'],axis='columns')
y=df5.Survived
x.Age=x.Age.fillna(x.Age.mean())
x.Fare=x.Fare.fillna(x.Fare.mean())
#Splitting dataset into train model
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.25,random_state=2)
#Load model and fit data into model
model=GaussianNB()
model.fit(X_train,Y_train)
#Dump fitted model into pickle file
pickle.dump(model, open(path + 'Titanic_survival_prediction_model.pkl', 'wb'))
print("Model 5 saved Succesfully")




#dataset for model 6
df6 = pd.read_csv("Datasets/Position_Salaries.csv")
# independent & dependent data
X = df6.iloc[:, :-1].values    # Features => Years of experience => Independent Variable
y = df6.iloc[:, -1].values     # Target => Salary => Dependent Variable
#Splitting dataset into train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)
#Load model(linear regression) and fit data into model
model=LinearRegression()
model.fit(X_train,y_train)
#Dump fitted model into pickle file
pickle.dump(model, open(path + 'Employee_salary_prediction_model.pkl', 'wb'))
print("Model 6 saved Succesfully")




#dataset for model 7
df7 = pd.read_csv("Datasets/purchase_pred.csv")
# independent & dependent data
X=df7.drop(['Response'],axis='columns')
Y=df7['Response']
#Splitting dataset into train model
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
#Load model(logistic regression) and fit data into model
model=LogisticRegression()
model.fit(X_train,Y_train)
#Dump fitted model into pickle file
pickle.dump(model, open(path + 'next_purchase_prediction_model.pkl', 'wb'))
print("Model 7 saved Succesfully")




# dataset for model 8
df8 = pd.read_csv("Datasets/house_data.csv")
# independent & dependent data
X = df8.drop(['price'],axis=1)
Y = df8['price']
#Splitting dataset into train model
x_train , x_test , y_train , y_test = train_test_split(X ,Y, test_size = 0.10,random_state =2)
#Load model(logistic regression) and fit data into model
model = LinearRegression()
model.fit(x_train,y_train)
#Dump fitted model into pickle file
pickle.dump(model, open(path + 'House_price_prediction_model.pkl', 'wb'))
print("Model 8 saved Succesfully")
