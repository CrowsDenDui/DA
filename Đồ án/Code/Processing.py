import numpy as np
import pandas as pd
import datetime
from datetime import date
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import warnings
warnings.filterwarnings('ignore')
data=pd.read_csv('DA\marketing_campaign.csv',header=0,sep=';')
# print(data)
#Tạo biến chi tiêu
data['Age']=2014-data['Year_Birth']

data['Spending']=data['MntWines']+data['MntFruits']+data['MntMeatProducts']+data['MntFishProducts']+data['MntSweetProducts']+data['MntGoldProds']
#Tạo biến thâm niên
last_date = date(2014,10, 4)
data['Seniority']=pd.to_datetime(data['Dt_Customer'], dayfirst=True,format = '%Y-%m-%d')
data['Seniority'] = pd.to_numeric(data['Seniority'].dt.date.apply(lambda x: (last_date - x)).dt.days, downcast='integer')/30
data=data.rename(columns={'NumWebPurchases': "Web",'NumCatalogPurchases':'Catalog','NumStorePurchases':'Store'})
data['Marital_Status']=data['Marital_Status'].replace({'Divorced':'Alone','Single':'Alone','Married':'In couple','Together':'In couple','Absurd':'Alone','Widow':'Alone','YOLO':'Alone'})
data['Education']=data['Education'].replace({'Basic':'Undergraduate','2n Cycle':'Undergraduate','Graduation':'Postgraduate','Master':'Postgraduate','PhD':'Postgraduate'})

data['Children']=data['Kidhome']+data['Teenhome']
data['Has_child'] = np.where(data.Children> 0, 'Has child', 'No child')
data['Children'].replace({3: "3 children",2:'2 children',1:'1 child',0:"No child"},inplace=True)
data=data.rename(columns={'MntWines': "Wines",'MntFruits':'Fruits','MntMeatProducts':'Meat','MntFishProducts':'Fish','MntSweetProducts':'Sweets','MntGoldProds':'Gold'})


data=data[['Age','Education','Marital_Status','Income','Spending','Seniority','Has_child','Children','Wines','Fruits','Meat','Fish','Sweets','Gold']]
data.head()
# loại bỏ các giá trị ngoại lệ và các giá trị còn thiếu trong tập dữ liệu
data=data.dropna(subset=['Income'])
data=data[data['Income']<600000]

print(data)
# Chuẩn hóa dữ liệu
# chuẩn hóa các biến như Income, Seniority và Spending, đưa chúng về cùng một khoảng giá trị
scaler=StandardScaler()
dataset_temp=data[['Income','Seniority','Spending']]
X_std=scaler.fit_transform(dataset_temp)
# norm='l2' là để đảm bảo rằng mỗi hàng của dữ liệu có tổng bình phương bằng 1
X = normalize(X_std,norm='l2')

# Huấn luyện mô hình Gaussian Mixture Model (GMM)
# 2 thành phần (clusters) và kiểu covariance là spherical để phân loại dữ liệu đã được chuẩn hoá
gmm=GaussianMixture(n_components=2, covariance_type='spherical',max_iter=2000, random_state=3).fit(X)
labels = gmm.predict(X)
# Gán nhãn cho từng mẫu dữ liệu dựa trên kết quả của GMM và thay thế các nhãn số thành nhãn thể hiện ý nghĩa, ví dụ như 'VIP' và 'Visiting guests
dataset_temp['Cluster'] = labels
dataset_temp=dataset_temp.replace({0:'VIP',1:'Visiting guests'})
# Kết hợp thông tin nhóm đã được gán với dữ liệu gốc thông qua các chỉ số của DataFrame
data = data.merge(dataset_temp.Cluster, left_index=True, right_index=True)

# Tính và hiển thị tóm tắt thống kê bao gồm các chỉ số như mean, std, min, max cho mỗi nhóm (Cluster) trên các biến Income, Spending và Seniority
pd.options.display.float_format = "{:.0f}".format
summary=data[['Income','Spending','Seniority','Cluster']]
summary.set_index("Cluster", inplace = True)
summary=summary.groupby('Cluster').describe().transpose()
summary.head()

print(summary)

# tạo ra một figure với kích thước 7x7 inches
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
# scatter 3D với các điểm dữ liệu từ từng nhóm
for cluster_label in data['Cluster'].unique():
    cluster_data = data[data['Cluster'] == cluster_label]
    ax.scatter(cluster_data['Income'], cluster_data['Seniority'], cluster_data['Spending'], label=str(cluster_label),
               s=6, linewidths=1)

ax.set_xlabel('Income')
ax.set_ylabel('Seniority')
ax.set_zlabel('Spending')
ax.set_title('Customer Segmentation')

plt.legend()
plt.show()