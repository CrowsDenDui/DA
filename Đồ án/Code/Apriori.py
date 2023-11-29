import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import time, os, psutil
import seaborn as sns
from datetime import date
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

warnings.filterwarnings('ignore')
data=pd.read_csv('DA\marketing_campaign.csv',header=0,sep=';')

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

# chuẩn hóa các biến như Income, Seniority và Spending, đưa chúng về cùng một khoảng giá trị
scaler=StandardScaler()
dataset_temp=data[['Income','Seniority','Spending']]
X_std=scaler.fit_transform(dataset_temp)

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
data.drop(columns=['Wines', 'Fruits', 'Meat', 'Fish', 'Sweets', 'Gold'], inplace=True)
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
start_time = time.time()  # Bắt đầu tính thời gian thuật toán chạy
frequent_items = apriori(df, use_colnames=True, min_support=min_support, max_len=max_len + 1)
end_time = time.time()  # Kết thúc thời điểm thuật toán dừng

# Tính thời gian chạy và thiêu thụ bộ nhớ của thuật toán
timecost = end_time - start_time
memocost = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
print("Thời gian thực hiện: {0}".format(timecost) + "s")
print("Bộ nhớ tiêu thụ: {0}".format(memocost) + "MB")

# Áp dụng thuật toán tạo luật kết hợp (association rules) dựa trên mục phổ biến
start_time = time.time()  # Bắt đầu tính thời gian thuật toán chạy
rules = association_rules(frequent_items, metric='lift', min_threshold=1)
end_time = time.time()  # Kết thúc thời điểm thuật toán dừng

# Tính thời gian chạy và thiêu thụ bộ nhớ của thuật toán
timecost = end_time - start_time
memocost = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

print("Thời gian thực hiện: {0}".format(timecost) + "s")
print("Bộ nhớ tiêu thụ: {0}".format(memocost) + "MB")

# Chọn sản phẩm và phân khúc cần kiểm tra
product='Wines'
segment='Biggest consumer'
target = '{\'%s_segment_%s\'}' %(product,segment)
# Lọc các luật kết hợp liên quan đến sản phẩm và phân khúc được chọn
results_personnal_care = rules[rules['consequents'].astype(str).str.contains(target, na=False)].sort_values(by='confidence', ascending=False)
print(results_personnal_care.head())

# In thông tin về khách hàng lớn nhất của rượu vang
largest_wine_consumer = data[data['Wines_segment'] == 'Biggest consumer'].head(1)
print("\nThông tin về khách hàng lớn nhất của rượu vang:")
print(largest_wine_consumer[['Income', 'Spending', 'Seniority', 'Education']])

# Vẽ đồ thị
plt.plot([timecost], [memocost], color='red', marker='o', linestyle='dashed', linewidth=2, markersize=8)
plt.xlabel('Thời gian thực hiện (second)')
plt.ylabel('Bộ nhớ tiêu tốn (MB)')
plt.title('Thời gian thực hiện và bộ nhớ tiêu tốn')
plt.show()