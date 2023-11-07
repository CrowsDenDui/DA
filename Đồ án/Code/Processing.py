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
# Ngôi sao: Khách hàng cũ có thu nhập cao và chi tiêu cao.
# Cần chú ý: Khách hàng mới có thu nhập dưới mức trung bình và mức chi tiêu thấp.
# Tiềm năng cao: Khách hàng mới có thu nhập cao và mức chi tiêu cao.
# Nhóm Rò rỉ: Khách hàng cũ có thu nhập dưới mức trung bình và mức chi tiêu thấp.
# chuẩn hóa dữ liệu và sau đó tôi sẽ tạo phân cụm khách hàng theo các số liệu được xác định ở trên
scaler=StandardScaler()
dataset_temp=data[['Income','Seniority','Spending']]
X_std=scaler.fit_transform(dataset_temp)
X = normalize(X_std,norm='l2')

gmm=GaussianMixture(n_components=4, covariance_type='spherical',max_iter=2000, random_state=5).fit(X)
labels = gmm.predict(X)
dataset_temp['Cluster'] = labels
dataset_temp=dataset_temp.replace({0:'Stars',1:'Need attention',2:'High potential',3:'Leaky bucket'})
data = data.merge(dataset_temp.Cluster, left_index=True, right_index=True)

pd.options.display.float_format = "{:.0f}".format
summary=data[['Income','Spending','Seniority','Cluster']]
summary.set_index("Cluster", inplace = True)
summary=summary.groupby('Cluster').describe().transpose()
summary.head()

print(summary)

# PLOT = go.Figure()
# for C in list(data.Cluster.unique()):
    

#     PLOT.add_trace(go.Scatter3d(x = data[data.Cluster == C]['Income'],
#                                 y = data[data.Cluster == C]['Seniority'],
#                                 z = data[data.Cluster == C]['Spending'],                        
#                                 mode = 'markers',marker_size = 6, marker_line_width = 1,
#                                 name = str(C)))
# PLOT.update_traces(hovertemplate='Income: %{x} <br>Seniority: %{y} <br>Spending: %{z}')

    
# PLOT.update_layout(width = 800, height = 800, autosize = True, showlegend = True,
#                    scene = dict(xaxis=dict(title = 'Income', titlefont_color = 'black'),
#                                 yaxis=dict(title = 'Seniority', titlefont_color = 'black'),
#                                 zaxis=dict(title = 'Spending', titlefont_color = 'black')),
#                    font = dict(family = "Gilroy", color  = 'black', size = 12))
# PLOT.show()