import numpy as np                               
import pandas as pd                               
import seaborn as sns
import datetime
from datetime import date
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing 
from math import sqrt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from apyori import apriori
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
import warnings 
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings('ignore')

#Load dữ liệu từ file
df = pd.read_csv('Doan\marketing_campaign.csv',header=0, sep="\t")

#Loại bỏ các cột dữ liệu không cần thiết
df = df.drop(['Z_CostContact', 'Z_Revenue'],axis=1)

#Xử lí dữ liệu trống trong cột income
df['Income'] = df['Income'].fillna(df['Income'].mean())
df.isna().any()

#Xử lí dữ liệu của "Marital_Status" từ phức tạp thành đơn giản
df['Marital_Status'] = df['Marital_Status'].replace(['Married', 'Together'],'relationship')
df['Marital_Status'] = df['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'],'Single')

#Tách các sản phẩm thành khung dữ liệu khác
product_data = []
for i in range(0, len(df)):
  productdata = [df['MntWines'][i], df['MntFruits'][i], 
                  df['MntMeatProducts'][i], df['MntFishProducts'][i], 
                  df['MntSweetProducts'][i], df['MntGoldProds'][i]]
  product_data.append(productdata)
Products_DF = pd.DataFrame(product_data, columns = ['Wines', 'Fruits', 'Meat', 'Fish', 'Sweets', 'Gold'])
Products_DF.head()

#Kết hợp các cột dữ liệu thành một duy nhất để loại bỏ dữ liệu thừa
df['Kids'] = df['Kidhome'] + df['Teenhome']
df['Expenses'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
df['TotalAcceptedCmp'] = df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3'] + df['AcceptedCmp4'] + df['AcceptedCmp5'] + df['Response']
df['NumTotalPurchases'] = df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases'] + df['NumDealsPurchases']

#Xoá một số cột để giảm kích thước và độ phức tạp của mô hình
col_del = ["AcceptedCmp1" , "AcceptedCmp2", "AcceptedCmp3" , "AcceptedCmp4","AcceptedCmp5", "Response","NumWebVisitsMonth", "NumWebPurchases","NumCatalogPurchases","NumStorePurchases","NumDealsPurchases" , "Kidhome", "Teenhome","MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
df=df.drop(columns=col_del,axis=1)

#Tạo cột "Age" cho DataFrame
df['Age'] = 2015 - df["Year_Birth"]

#Đổi dữ liệu cột "Education" thành "UG" (Undergradute) và "PG" (Postgraduate)
df['Education'] = df['Education'].replace(['PhD','2n Cycle','Graduation', 'Master'],'PG')  
df['Education'] = df['Education'].replace(['Basic'], 'UG')

#Đổi dữ liệu cột "Dt_Customer" thành foramt timestamp
df['Dt_Customer'] = pd.to_datetime(df.Dt_Customer, format = '%d-%m-%Y')
df['first_day'] = '01-01-2015'
df['first_day'] = pd.to_datetime(df.first_day)
df['day_engaged'] = (df['first_day'] - df['Dt_Customer']).dt.days

df=df.drop(columns=["ID", "Dt_Customer", "first_day", "Year_Birth", "Dt_Customer", "Recency", "Complain"],axis=1)

print(df.head())

#Mô hình dữ liệu số lượng theo "Marital_Status"
sns.countplot(df.Marital_Status)
sns.set(rc={'figure.figsize':(4,4)})
plt.show()

#Cấu hình lại mô hình
plt.rcParams.update(plt.rcParamsDefault)

#Mô hình phân tích mối quan hệ giữa "Marital_Status" và "Expenses" với "Education"
plt.figure(figsize=(8,8))
sns.barplot(x=df['Marital_Status'], y=df['Expenses'], hue = df["Education"])
plt.title("Analysis of the Correlation between Marital Status and Expenses with respect to Education")

#Mô hình phân tích mối quan hệ giữa "Marital_Status" với "Expenses"
plt.figure(figsize=(8,8))
sns.barplot(x=df['Marital_Status'], y=df['Expenses'])
plt.title("Analysis of the Correlation between Marital Status and Expenses")

#Mô hình phân phối "Expenses" theo "Marital_Status"
plt.figure(figsize=(8,8))
plt.hist("Expenses", data = df[df["Marital_Status"] == "relationship"], alpha = 0.5, label = "relationship")
plt.hist("Expenses", data = df[df["Marital_Status"] == "Single"], alpha = 0.5, label = "Single")
plt.title("Distribution of Expenses with respect to Marital Status")
plt.xlabel("Expenses")
plt.legend(title = "Marital Status")

#Mô hình phân phối "Expenses" theo "Education"
plt.figure(figsize=(8,8))
plt.hist("Expenses", data = df[df["Education"] == "PG"], alpha = 0.5, label = "PG")
plt.hist("Expenses", data = df[df["Education"] == "UG"], alpha = 0.5, label = "UG")
plt.title("Distribution of Expenses with respect to Education")
plt.xlabel("Expenses")
plt.legend(title = "Education")

#Mô hình phân phối "NumTotalPurchases" với "Education"
plt.figure(figsize=(8,8))
plt.hist("NumTotalPurchases", data = df[df["Education"] == "PG"], alpha = 0.5, label = "PG")
plt.hist("NumTotalPurchases", data = df[df["Education"] == "UG"], alpha = 0.5, label = "UG")
plt.title("Distribution of Number of Total Purchases with respect to Education")
plt.xlabel("Number of Total Purchases")
plt.legend(title = "Education")

#Mô hình phân bổ "Age" theo "Marital_Status"
plt.figure(figsize=(8,8))
plt.hist("Age", data = df[df["Marital_Status"] == "relationship"], alpha = 0.5, label = "relationship")
plt.hist("Age", data = df[df["Marital_Status"] == "Single"], alpha = 0.5, label = "Single")
plt.title("Distribution of Age with respect to Marital Status")
plt.xlabel("Age")
plt.legend(title = "Marital Status")

#Mô hình phân phối "Income" theo "Marital_Status"
plt.figure(figsize=(8,8))
plt.hist("Income", data = df[df["Marital_Status"] == "relationship"], alpha = 0.5, label = "relationship")
plt.hist("Income", data = df[df["Marital_Status"] == "Single"], alpha = 0.5, label = "Single")
plt.title("Distribution of Income with respect to Marital Status")
plt.xlabel("Income")
plt.legend(title = "Marital Status")

#Mô hình phân tích số lượng khách hàng theo "Marital_Status"
plt.figure(figsize=(8,8))
plt.pie(df["Marital_Status"].value_counts(), labels = ["relationship", "Single"], autopct='%1.1f%%', counterclock=False)
plt.legend()

#Mô hình phân tích số lượng khách hàng theo "Education"
plt.figure(figsize=(8,8))
plt.pie(df["Education"].value_counts(), labels = ["PG", "UG"], autopct='%1.1f%%', counterclock=False)
plt.legend()

#Mô hình phân phối "Expenses" dựa trên "Education"
sns.barplot(x = df['Expenses'],y = df['Education']);
plt.title('Total Expense based on the Education Level');

#Mô hình dữ liệu "Income" dựa trên cấp độ "Education"
sns.barplot(x = df['Income'],y = df['Education']);
plt.title('Total Income based on the Education Level');

plt.show()

#Tạo danh sách các cột có dữ liệu kiểu "object"
cate = []
for i in df.columns:
    if (df[i].dtypes == "object"):
        cate.append(i)

print(cate)

#Chuyển kiểu dữ liệu "object" thành số
lbl_encode = LabelEncoder()
for i in cate:
    df[i]=df[[i]].apply(lbl_encode.fit_transform)

df1 = df.copy()

#Mô hình Boxplot để kiểm tra ngoại lệ
plt.figure(figsize=(5,5))
ax = sns.boxplot(data=df1 , orient="h")
plt.title('A boxplot: Outliers in the dataset', color = 'blue')
plt.xlabel('Count/ Frequency')
plt.show()

#Loại bỏ các ngoại lệ
q3 = df1.quantile(0.75)
q1 = df1.quantile(0.25)
iqr = q3-q1
lower_range = q1 - (1.5 * iqr)
upper_range = q3 + (1.5 * iqr)

df1 = df1[~( (df1 < lower_range)|(df1 > upper_range) ).any(axis=1)]

#Xây dựng và vẽ đồ thị Elbow method để chọn số clusters
wcss=[] 
for i in range (1,11): 
 kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
 kmeans.fit(df1)
 wcss.append(kmeans.inertia_)
plt.figure(figsize=(16,8))
plt.plot(range(1,11),wcss, 'bx-')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Xây dựng và vẽ đồ thị Silhouette method để chọn số clusters
silhouette_scores = []
for i in range(2,10):
    m1=KMeans(n_clusters=i, random_state=42)
    c = m1.fit_predict(df1)
    silhouette_scores.append(silhouette_score(df1, m1.fit_predict(df1))) 
plt.bar(range(2,10), silhouette_scores) 
plt.xlabel('Number of clusters', fontsize = 20) 
plt.ylabel('S(i)', fontsize = 20) 
plt.show()

#Sử dụng Silhouette để đo giá trị của K
# print(silhouette_scores)

#Lấy giá trị tối đa của điểm hình bóng và thêm 2 vào chỉ mục vì chỉ mục bắt đầu từ 2
sc=max(silhouette_scores)
number_of_clusters=silhouette_scores.index(sc)+2
print("Number of Cluster Required is : ", number_of_clusters)

#Huấn luyện dự đoán bằng Thuật toán K-Means.
kmeans=KMeans(n_clusters=number_of_clusters, random_state=42).fit(df1)
pred=kmeans.predict(df1)

#Nối các giá trị cụm đó vào khung dữ liệu chính (không có vô hướng chuẩn)
df1['cluster'] = pred + 1

pl = sns.countplot(x=df1["cluster"])
pl.set_title("Distribution Of The Clusters")
plt.show()
sns.set(rc={'axes.facecolor':'gray', 'figure.facecolor':'gray', 'axes.grid' : False, 'font.family': 'Arial'})

#Vẽ đồ thị phân phối các features theo clusters
for i in df1:
    diag = sns.FacetGrid(df1, col = "cluster", hue = "cluster", palette = "Set1")
    diag.map(plt.hist, i, bins=6, ec="k") 
    diag.set_xticklabels(rotation=25, color = 'white')
    diag.set_yticklabels(color = 'white')
    diag.set_xlabels(size=16, color = 'white')
    diag.set_titles(size=16, color = '#FFFF00', fontweight="bold")
    diag.fig.set_figheight(6)

#Thuật toán Apriori
data = df1.copy()

#Tạo "Age" Segment
cut_labels_Age = ['Young', 'Adult', 'Mature', 'Senior']
cut_bins = [0, 30, 45, 65, 120]
data['Age_group'] = pd.cut(data['Age'], bins=cut_bins, labels=cut_labels_Age)

#Tạo "Income" segment
cut_labels_Income = ['Low income', 'Low to medium income', 'Medium to high income', 'High income']
data['Income_group'] = pd.qcut(data['Income'], q=4, labels=cut_labels_Income)

#Tạo "Day engaged" segment
cut_labels_dayengaged = ['New customers', 'Discovering customers', 'Experienced customers', 'Old customers']
data['dayengaged_group'] = pd.qcut(data['day_engaged'], q=4, labels=cut_labels_dayengaged)
data=data.drop(columns=['Age','Income','day_engaged'])

#Tạo nhóm cho các sản phẩm dựa trên giá trị và phân vị
cut_labels = ['Least Active Customer', 'Highly Active Customer']
data['Wines_segment'] = pd.qcut(Products_DF['Wines'][Products_DF['Wines']>0],q=[0, 0.5 ,1], labels=cut_labels).astype("object")
data['Fruits_segment'] = pd.qcut(Products_DF['Fruits'][Products_DF['Fruits']>0],q=[0, 0.5, 1], labels=cut_labels).astype("object")
data['Meat_segment'] = pd.qcut(Products_DF['Meat'][Products_DF['Meat']>0],q=[0, 0.5,1], labels=cut_labels).astype("object")
data['Fish_segment'] = pd.qcut(Products_DF['Fish'][Products_DF['Fish']>0],q=[0, 0.5, 1], labels=cut_labels).astype("object")
data['Sweets_segment'] = pd.qcut(Products_DF['Sweets'][Products_DF['Sweets']>0],q=[0, 0.5, 1], labels=cut_labels).astype("object")
data['Gold_segment'] = pd.qcut(Products_DF['Gold'][Products_DF['Gold']>0],q=[0, 0.5, 1], labels=cut_labels).astype("object")

#Thay thế các giá trị "NaN" thành "Inactive Customer"
data.replace(np.nan, "Inactive Customer",inplace=True)
data = data.astype(object)

print(data.head())

#Cài đặt hiển thị tất cả các cột và bảng trong DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 999)
pd.options.display.float_format = "{:.3f}".format

association = data.copy() 
association.head()

association.drop(["Education", "Marital_Status", "Kids", "Expenses", "TotalAcceptedCmp", "NumTotalPurchases", "cluster"], axis = 1, inplace = True)
association.head()

#Áp dụng thuật toán Apriori để tính các mục phổ biến
df_ap = pd.get_dummies(association)
min_support = 0.08
max_len = 10
frequent_items = apriori(df_ap, use_colnames=True, min_support=min_support, max_len=max_len + 1)
rules = association_rules(frequent_items, metric='lift', min_threshold=1)

#Phân tích quy tắc kết hợp cho sản phẩm "Wines" với đối tượng "Highly Active Customer"
product='Wines'
segment='Highly Active Customer'
target = '{\'%s_segment_%s\'}' %(product,segment)
results_personnel_care = rules[rules['consequents'].astype(str).str.contains(target, na=False)].sort_values(by='confidence', ascending=False)
print(results_personnel_care.head())

#Phân tích quy tắc kết hợp cho sản phẩm "Fruits" với đối tượng "Highly Active Customer"
product='Fruits'
segment='Highly Active Customer'
target = '{\'%s_segment_%s\'}' %(product,segment)
results_personnel_care = rules[rules['consequents'].astype(str).str.contains(target, na=False)].sort_values(by='confidence', ascending=False)
print(results_personnel_care.head())

#Phân tích quy tắc kết hợp cho sản phẩm "Meat" với đối tượng "Highly Active Customer"
product='Meat'
segment='Highly Active Customer'
target = '{\'%s_segment_%s\'}' %(product,segment)
results_personnel_care = rules[rules['consequents'].astype(str).str.contains(target, na=False)].sort_values(by='confidence', ascending=False)
print(results_personnel_care.head())

#Phân tích quy tắc kết hợp cho sản phẩm "Fish" với đối tượng "Highly Active Customer"
product='Fish'
segment='Highly Active Customer'
target = '{\'%s_segment_%s\'}' %(product,segment)
results_personnel_care = rules[rules['consequents'].astype(str).str.contains(target, na=False)].sort_values(by='confidence', ascending=False)
print(results_personnel_care.head())

#Phân tích quy tắc kết hợp cho sản phẩm "Sweets" với đối tượng "Highly Active Customer"
product='Sweets'
segment='Highly Active Customer'
target = '{\'%s_segment_%s\'}' %(product,segment)
results_personnel_care = rules[rules['consequents'].astype(str).str.contains(target, na=False)].sort_values(by='confidence', ascending=False)
print(results_personnel_care.head())

#Phân tích quy tắc kết hợp cho sản phẩm "Gold" với đối tượng "Highly Active Customer"
product='Gold'
segment='Highly Active Customer'
target = '{\'%s_segment_%s\'}' %(product,segment)
results_personnel_care = rules[rules['consequents'].astype(str).str.contains(target, na=False)].sort_values(by='confidence', ascending=False)
print(results_personnel_care.head())

df2=df1.copy()
x = df2.drop('cluster', axis=1)
y = df2['cluster']

#Chia dữ liệu thành tập huấn luyện và tập kiểm tra
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

#Chuẩn hoá dữ liệu bằng StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Xây dựng mô hình Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

#Dự đoán trên tập kiểm tra
y_predicted = log_reg.predict(x_test)

#In báo cáo các phân loại
print("Classification Report: \n", classification_report(y_test,y_predicted))
print("-" * 100)
print()
 
#Độ chính xác   
acc = accuracy_score(y_test, y_predicted)

print("Accuracy Score: ", acc)
print("-" * 100)
print()

#Điểm f1
f1 = f1_score(y_test, y_predicted)

print("F1 Score: ", f1)
print("-" * 100)
print()

#Ma trận hỗn loạn 
print("Confusion Matrix: ")
plt.figure(figsize=(10, 5))
sns.heatmap(confusion_matrix(y_test, y_predicted), annot=True, fmt='g');
plt.title('Confusion Matrix', fontsize=20)
plt.show()