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

# chuẩn bị dữ liệu cho thuật toán Apriori
# phân khúc khách hàng theo độ tuổi, thu nhập và thâm niên
# tạo phân khúc Độ tuổi

cut_labels_Age = ['Young', 'Adult', 'Mature', 'Senior']
cut_bins = [0, 30, 45, 65, 120]
data['Age_group'] = pd.cut(data['Age'], bins=cut_bins, labels=cut_labels_Age)
# tạo phân khúc thu nhập
cut_labels_Income = ['Low income', 'Low to medium income', 'Medium to high income', 'High income']
data['Income_group'] = pd.qcut(data['Income'], q=4, labels=cut_labels_Income)
# tạo phân khúc thâm niên
cut_labels_Seniority = ['New customers', 'Discovering customers', 'Experienced customers', 'Old customers']
data['Seniority_group'] = pd.qcut(data['Seniority'], q=4, labels=cut_labels_Seniority)
data=data.drop(columns=['Age','Income','Seniority'])

# xác định các phân khúc mới theo mức chi tiêu của khách hàng cho từng sản phẩm
# Không phải người mua
# Người mua thấp
# Người mua thường xuyên
# Người mua lớn nhất

# Tạo các nhóm cho các mức tiêu thụ của sản phẩm từ 'Low consumer' đến 'Biggest consumer'
cut_labels = ['Low consumer', 'Frequent consumer', 'Biggest consumer']
# Tạo cột 'Wines_segment' với nhãn tương ứng cho mức tiêu thụ rượu
data['Wines_segment'] = pd.qcut(data['Wines'][data['Wines'] > 0], q=[0, .25, .75, 1], labels=cut_labels).astype("object")
# Tạo cột 'Fruits_segment' với nhãn tương ứng cho mức tiêu thụ trái cây
data['Fruits_segment'] = pd.qcut(data['Fruits'][data['Fruits'] > 0], q=[0, .25, .75, 1], labels=cut_labels).astype("object")
# Tạo cột 'Meat_segment' với nhãn tương ứng cho mức tiêu thụ thịt
data['Meat_segment'] = pd.qcut(data['Meat'][data['Meat'] > 0], q=[0, .25, .75, 1], labels=cut_labels).astype("object")
# Tạo cột 'Fish_segment' với nhãn tương ứng cho mức tiêu thụ cá
data['Fish_segment'] = pd.qcut(data['Fish'][data['Fish'] > 0], q=[0, .25, .75, 1], labels=cut_labels).astype("object")
# Tạo cột 'Sweets_segment' với nhãn tương ứng cho mức tiêu thụ đồ ngọt
data['Sweets_segment'] = pd.qcut(data['Sweets'][data['Sweets'] > 0], q=[0, .25, .75, 1], labels=cut_labels).astype("object")
# Tạo cột 'Gold_segment' với nhãn tương ứng cho mức tiêu thụ vàng bạc
data['Gold_segment'] = pd.qcut(data['Gold'][data['Gold'] > 0], q=[0, .25, .75, 1], labels=cut_labels).astype("object")
# Thay thế giá trị NaN bằng "Non consumer"
data.replace(np.nan, "Non consumer", inplace=True)
# Loại bỏ các cột không cần thiết liên quan đến chi tiêu
data.drop(columns=['Spending', 'Wines', 'Fruits', 'Meat', 'Fish', 'Sweets', 'Gold'], inplace=True)
# Chuyển đổi toàn bộ DataFrame thành kiểu dữ liệu object
data = data.astype(object)

# Thiết lập hiển thị tối đa cho cột, hàng, và độ rộng cột trong DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 999)

# Định dạng hiển thị số thập phân trong DataFrame
pd.options.display.float_format = "{:.3f}".format
# Tạo DataFrame 'association' là bản sao của 'data'
association = data.copy()
# Tạo one-hot encoding cho các biến phân loại trong DataFrame
df = pd.get_dummies(association)
# Thiết lập giá trị support tối thiểu và chiều dài tối đa cho luật kết hợp
min_support = 0.08
max_len = 10
# Áp dụng thuật toán Apriori để tìm các mục phổ biến
frequent_items = apriori(df, use_colnames=True, min_support=min_support, max_len=max_len + 1)
# Áp dụng thuật toán tạo luật kết hợp (association rules) dựa trên mục phổ biến
rules = association_rules(frequent_items, metric='lift', min_threshold=1)
# Xác định sản phẩm và segment cụ thể cần quan tâm
product = 'Wines'
segment = 'Biggest consumer'
target = '{\'%s_segment_%s\'}' % (product, segment)
# Lọc các luật kết hợp liên quan đến sản phẩm và segment cụ thể
results_personnal_care = rules[rules['consequents'].astype(str).str.contains(target, na=False)].sort_values(by='confidence', ascending=False)
results_personnal_care.head()
# in ra kết quả
print(association)