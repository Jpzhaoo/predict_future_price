# 0.0 加载库和默认设定
# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import category_encoders as ce
import warnings
from collections import Counter

pd.set_option('display.max_columns', 160)
pd.set_option('display.max_rows', 400)
pd.set_option('display.max_colwidth', 40)
warnings.filterwarnings('ignore')


# 1.0 数据集准备 ---------------------------
test = pd.read_csv("./data/test.csv")
test.head()

## 1.1 Item/category 信息 --------------


# %%
#------------------------------
# category 数据
categories = pd.read_csv("./data_eng/categories.csv")
categories.head()
(
    pd.DataFrame(categories
                 .category_name 
                 .values.reshape(-1,4)) # reshape(-1，4)代表行未知，4列
)
    # 可以看到大部分数据均含有 “-“，"-"前面的数据为主类别，后面的为子类别
categories['group_name'] = categories['category_name'].str.extract(r'(^[\w\s]*)')
categories['group_name'] = categories['group_name'].str.strip() # 去掉开头结尾的空格
# label encode group names 一共17个类别
categories['group_id']  = le.fit_transform(categories.group_name.values)
categories.sample(5)
#------------------------------

# %%
#------------------------------
# items 数据集
items = pd.read_csv("./data_eng/items.csv")
#clean item_name
# 首先lower, 接着去除以符号和特定单词开头的item_name, 去除所有的空格 
items['item_name'] = items['item_name'].str.lower()
items['item_name'] = items['item_name'].str.replace('.', '')
for i in [r'[^\w\d\s\.]', r'\bthe\b', r'\bin\b', r'\bis\b',
          r'\bfor\b', r'\bof\b', r'\bon\b', r'\band\b',  
          r'\bto\b', r'\bwith\b' , r'\byo\b']:
    items['item_name'] = items['item_name'].str.replace(i, ' ')
items['item_name'] = items['item_name'].str.replace(r'\b.\b', ' ')

# 提取前n个字符，
items['item_name_no_space'] = items['item_name'].str.replace(' ', '')
items['item_name_first4'] = [x[:4] for x in items['item_name_no_space']]
items['item_name_first6'] = [x[:6] for x in items['item_name_no_space']]
items['item_name_first11'] = [x[:11] for x in items['item_name_no_space']]
del items['item_name_no_space']
                              
# 进行label 编码
items.item_name_first4 = le.fit_transform(items.item_name_first4.values)
items.item_name_first6 = le.fit_transform(items.item_name_first6.values)
items.item_name_first11 = le.fit_transform(items.item_name_first11.values)

# 连接 items 表和 categories 表
items = items.join(categories.set_index('category_id'), on='category_id')
items.sample(10)
#------------------------------



# %%
#------------------------------
# 重复数据
# items 中存在重复的行，因此创建字典
# 根据 item_name, category_id 判断数据是否重复
dupes = items[(items.duplicated(subset=['item_name','category_id'], keep=False))] # keep = False重复数据均标记为 True
# dupes.sort_values(['item_id','category_id'])
# 重复数据 item_id 是否在测试集中
dupes['in_test'] = dupes.item_id.isin(test.item_id.unique())
# 
dupes = dupes.groupby('item_name').agg({'item_id':['first','last'],'in_test':['first','last']})

# 目的是让训练集中的 item_id 更多的符合测试集的 item_id
# 如果 first 和 last item_id 均在测试集， 则不处理
dupes = dupes[(dupes[('in_test', 'first')]==False) | (dupes[('in_test', 'last')]==False)]
# 如果仅仅第一个 item_id 在测试集，则last 也变为 first
temp = dupes[dupes[('in_test', 'first')]==True]
keep_first = dict(zip(temp[('item_id', 'last')], temp[('item_id',  'first')]))
# 如果两者都在或者只有 last item_id 在测试集， 则将first 变为 last
temp = dupes[dupes[('in_test', 'first')]==False]
keep_second = dict(zip(temp[('item_id', 'first')], temp[('item_id',  'last')]))
item_map = {**keep_first, **keep_second}
#------------------------------


## 1.2 Sales 信息----------------
# %%
#------------------------------
#loading sales data
from plotnine import *
import seaborn as sns
import matplotlib.pyplot as plt
sales = pd.read_csv('./data/sales_train.csv')
# 有60个shop_id
sorted(sales.shop_id.unique())
sns.boxplot(sales['item_price'],
            orient='v')
# 查看异常值
# sales.query('item_price > 300000')
# items.query('item_id == 6066')
sns.boxplot(sales['item_cnt_day'],
            orient='v')
sns.countplot(sales['shop_id'])
sales[sales.duplicated(['item_id'], keep=False)]

# 由于item_id 有大量的重复，根据前面的方法尽可能的将 item_id 改为测试集中的 item_id
sales = (sales
    .query('0 < item_price < 50000 and 0 < item_cnt_day < 1001') # 去除异常值
    .replace({
        'shop_id':{0:57, 1:58, 11:10}, #replacing obsolete shop id's
        'item_id':item_map # fixing duplicate item id's  
    })    
)

# 移除测试集中没有的shop 
sales = sales[sales['shop_id'].isin(test.shop_id.unique())]

sales['date'] = pd.to_datetime(sales.date, format='%d.%m.%Y')
sales['weekday'] = sales.date.dt.dayofweek

# 某商品第一次售出的时间, 0为训练集的起始日期
sales['first_sale_day'] = sales.date.dt.dayofyear 
sales['first_sale_day'] += 365 * (sales.date.dt.year-2013)
sales['first_sale_day'] = sales.groupby('item_id')['first_sale_day'].transform('min').astype('int16')

# 计算总销售额
sales['revenue'] = sales['item_cnt_day']*sales['item_price']
#------------------------------



# %%
#------------------------------
# 每个商店工作日销售量情况（百分比）

# 先计算每个商店工作日销售量
temp = sales.groupby(['shop_id','weekday']).agg({'item_cnt_day':'sum'}).reset_index()
# 再计算每个商店总的销售量
temp = pd.merge(temp, sales.groupby(['shop_id']).agg({'item_cnt_day':'sum'}).reset_index(), on='shop_id', how='left')
temp.columns = ['shop_id','weekday', 'shop_day_sales', 'shop_total_sales']
temp['day_quality'] = temp['shop_day_sales']/temp['shop_total_sales']
temp = temp[['shop_id','weekday','day_quality']]


dates = pd.DataFrame(data={'date':pd.date_range(start='2013-01-01',end='2015-11-30')})
dates['weekday'] = dates.date.dt.dayofweek
dates['month'] = dates.date.dt.month
dates['year'] = dates.date.dt.year - 2013
dates['date_block_num'] = dates['year']*12 + dates['month'] - 1
dates['first_day_of_month'] = dates.date.dt.dayofyear
dates['first_day_of_month'] += 365 * dates['year']
dates = dates.join(temp.set_index('weekday'), on='weekday')
# 月销售情况。其中 day_quality 用于衡量当月的销售能力，对于2月，由于时间跨度中不包含闰年，因此二月始终为4
dates = dates.groupby(['date_block_num','shop_id','month','year']).agg({'day_quality':'sum','first_day_of_month':'min'}).reset_index()

dates.query('shop_id == 28').head(15)

#------------------------------



# %%
#------------------------------
# 月销量
sales = (sales
     .groupby(['date_block_num', 'shop_id', 'item_id'])
     .agg({
         'item_cnt_day':'sum', 
         'revenue':'sum',
         'first_sale_day':'first'
     })
     .reset_index()
     .rename(columns={'item_cnt_day':'item_cnt'})
)
sales.sample(5)
#------------------------------



## 1.3 构造训练数据
# %%
#------------------------------
from itertools import product
# product 用于求笛卡尔乘积
df = [] 
for block_num in sales['date_block_num'].unique():
    cur_shops = sales.loc[sales['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = sales.loc[sales['date_block_num'] == block_num, 'item_id'].unique()
    df.append(np.array(list(product(*[cur_shops, cur_items, [block_num]]))))
# 垂直合并
df = pd.DataFrame(np.vstack(df), columns=['shop_id', 'item_id', 'date_block_num'])
df.head()

# 给测试集添加合适的 date_block_num
test['date_block_num'] = 34
del test['ID']

# append test set to training dataframe
df.isnull().sum() # 没有缺失值
df = pd.concat([df,test]).fillna(0)
df = df.reset_index()
del df['index']

#join sales and item inforamtion to the training dataframe
df = pd.merge(df, sales, on=['shop_id', 'item_id', 'date_block_num'], how='left').fillna(0)
df = pd.merge(df, dates, on=['date_block_num','shop_id'], how='left')
df = pd.merge(df, items.drop(columns=['item_name','group_name','category_name']), on='item_id', how='left')

#------------------------------

## 1.4 添加商店信息（shop_cluster, shop_type, shop_city)
# %%
#------------------------------
# 根据每个商店的不同种类商品的占比，对商店进行聚类。
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


shops = pd.read_csv('./data_eng/shops.csv')

# clustering shops
shops_cats = pd.DataFrame(
    np.array(list(product(*[df['shop_id'].unique(), df['category_id'].unique()]))),
    columns =['shop_id', 'category_id']
)
# 每个商店中不同种类的商品占该商店总商品数的比例
temp = df.groupby(['category_id', 'shop_id']).agg({'item_cnt':'sum'}).reset_index()
temp2 = temp.groupby('shop_id').agg({'item_cnt':'sum'}).rename(columns={'item_cnt':'shop_total'})
temp = temp.join(temp2, on='shop_id')
temp['category_proportion'] = temp['item_cnt']/temp['shop_total']
temp = temp[['shop_id', 'category_id', 'category_proportion']]
shops_cats = pd.merge(shops_cats, temp, on=['shop_id','category_id'], how='left')
shops_cats = shops_cats.fillna(0)

# 长变宽
shops_cats = shops_cats.pivot(index='shop_id', columns=['category_id'])

# K-means
inertia = []

for n in range(1 , 11):
    algorithm = (KMeans(n_clusters = n , n_init = 10, max_iter=300, 
                        random_state= 111) )
    algorithm.fit(shops_cats)
    inertia.append(algorithm.inertia_)
    cluster_labels = algorithm.labels_
    # 轮廓系数
    # silhouette_avg = silhouette_score(shops_cats, cluster_labels)
    # print("For n_clusters={0}, the silhouette score is {1}".format(n, silhouette_avg))
plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()

kmeans = KMeans(n_clusters=7, random_state=0).fit(shops_cats)
shops_cats['shop_cluster'] = kmeans.labels_.astype('int8')

#adding these clusters to the shops dataframe
# 由于shops 共有60个 shop_id, 而shops_cats 中只有42个，因此会有缺失值。
shops = shops.join(shops_cats['shop_cluster'], on='shop_id')
#removing unused shop ids
shops.dropna(inplace=True)
#------------------------------


# %%
#------------------------------
# 增加 shop_type, shop_city 特征
#cleaning the name column
shops['shop_name'] = shops['shop_name'].str.lower()
shops['shop_name'] = shops['shop_name'].str.replace(r'[^\w\d\s]', ' ')

#creating a column for the type of shop
shops['shop_type'] = 'regular'

#there is some overlap in tc and mall, mall is given precedence
shops.loc[shops['shop_name'].str.contains(r'tc'), 'shop_type'] = 'tc'
shops.loc[shops['shop_name'].str.contains(r'mall|center|mega'), 'shop_type'] = 'mall'
shops.loc[shops['shop_id'].isin([9,20]), 'shop_type'] = 'special'
shops.loc[shops['shop_id'].isin([12,55]), 'shop_type'] = 'online'

#the first word of shop name is largely sufficient as a city feature
shops['shop_city'] = shops['shop_name'].str.split().str[0]
shops.loc[shops['shop_id'].isin([12,55]), 'shop_city'] = 'online'
shops.shop_city = le.fit_transform(shops.shop_city.values)
shops.shop_type = le.fit_transform(shops.shop_type.values)
shops.head()

df = pd.merge(df, shops.drop(columns='shop_name'), on='shop_id', how='left')
df.head()
#------------------------------



## 1.5 item ages 以及销售和价格信息
# %%
#------------------------------
# 构造新的变量——某个item 前一次售出的时间距离当前月份的第一天的时长
# 某件商品在不同商店最晚卖出的时间
df['first_sale_day'] = df.groupby('item_id')['first_sale_day'].transform('max').astype('int16')
df.loc[df['first_sale_day']==0, 'first_sale_day'] = 1035 #  为0说明该商品还没有售出
df['prev_days_on_sale'] = [max(idx) for idx in zip(df['first_day_of_month']-df['first_sale_day'], [0]*len(df))] # 为0说明截止本月，该商品还没有卖出
del df['first_day_of_month']

#freeing RAM, removing unneeded columns and encoding object columns
del sales, categories, shops, shops_cats, temp, temp2, test, dupes, item_map, 
df.head()

df['item_cnt_unclipped'] = df['item_cnt']
# .999 分位数为21
df['item_cnt_unclipped'].quantile(0.999)
df['item_cnt'] = df['item_cnt'].clip(0, 20) # 下界为0，上界为20 超过20的替换为20，小于0的替换为0
#------------------------------

# %%
# 压缩数据，减小内存占用
#------------------------------
def downcast(df):
    #reduce size of the dataframe
    float_cols = [c for c in df if df[c].dtype in ["float64"]]
    int_cols = [c for c in df if df[c].dtype in ['int64']]
    df[float_cols] = df[float_cols].astype('float32')
    df[int_cols] = df[int_cols].astype('int16')
    return df
df = downcast(df)
df.info()
#------------------------------

# %%
#------------------------------
# df.query("shop_id == 45 & item_id == 18454")
# 当前月份减去商品/类别/组别/商店第一次出现的日期。
df['item_age'] = (df['date_block_num'] - df.groupby('item_id')['date_block_num'].transform('min')).astype('int8')
df['item_name_first4_age'] = (df['date_block_num'] - df.groupby('item_name_first4')['date_block_num'].transform('min')).astype('int8')
df['item_name_first6_age'] = (df['date_block_num'] - df.groupby('item_name_first6')['date_block_num'].transform('min')).astype('int8')
df['item_name_first11_age'] = (df['date_block_num'] - df.groupby('item_name_first11')['date_block_num'].transform('min')).astype('int8')
df['category_age'] = (df['date_block_num'] - df.groupby('category_id')['date_block_num'].transform('min')).astype('int8')
df['group_age'] = (df['date_block_num'] - df.groupby('group_id')['date_block_num'].transform('min')).astype('int8')
df['shop_age'] = (df['date_block_num'] - df.groupby('shop_id')['date_block_num'].transform('min')).astype('int8')
#------------------------------


# %%
#------------------------------

# 判断某件商品在某个商店之前是否卖出过
temp = df.query('item_cnt > 0').groupby(['item_id','shop_id']).agg({'date_block_num':'min'}).reset_index()
temp.columns = ['item_id', 'shop_id', 'item_shop_first_sale']
temp.item_shop_first_sale.values.max() # 33
df = pd.merge(df, temp, on=['item_id','shop_id'], how='left')
# 未售出的商品，将其第一次售出月份填充为50
df['item_shop_first_sale'] = df['item_shop_first_sale'].fillna(50)
# 如果某件商品在某个商店已售出（第一次），item_age_if_shop_sale 则为 item_age
df['item_age_if_shop_sale'] = (df['date_block_num'] > df['item_shop_first_sale']) * df['item_age']
# 某件商品从开始出售的 age （还没售出）的item_age
df['item_age_without_shop_sale'] = (df['date_block_num'] <= df['item_shop_first_sale']) * df['item_age']
del df['item_shop_first_sale']
#------------------------------


# %% 
#------------------------------
# 根据不同的分组汇总 item_cnt 平均数
def agg_cnt_col(df, merging_cols, new_col, aggregation):
    temp = df.groupby(merging_cols).agg(aggregation).reset_index()
    temp.columns = merging_cols + [new_col]
    df = pd.merge(df, temp, on=merging_cols, how='left')
    return df

#individual items across all shops
# 当月该 item 在每家商店平均售出量
df = agg_cnt_col(df, ['date_block_num','item_id'],'item_cnt_all_shops',{'item_cnt':'mean'})
df = agg_cnt_col(df, ['date_block_num','item_id'],'item_cnt_all_shops_median',{'item_cnt':'median'}) 
# 当月某种类在某商店平均/中位数销量
df = agg_cnt_col(df, ['date_block_num','category_id','shop_id'],'category_cnt',{'item_cnt':'mean'})
df = agg_cnt_col(df, ['date_block_num','category_id','shop_id'],'category_cnt_median',{'item_cnt':'median'}) 
# 当月某种类在所有商店平均/中位数销量
df = agg_cnt_col(df, ['date_block_num','category_id'],'category_cnt_all_shops',{'item_cnt':'mean'})
df = agg_cnt_col(df, ['date_block_num','category_id'],'category_cnt_all_shops_median',{'item_cnt':'median'})
# 当月某group在某商店平均销量
df = agg_cnt_col(df, ['date_block_num','group_id','shop_id'],'group_cnt',{'item_cnt':'mean'})
# 当月某group在所有商店平均销量
df = agg_cnt_col(df, ['date_block_num','group_id'],'group_cnt_all_shops',{'item_cnt':'mean'})
# 当月某商店平均销量
df = agg_cnt_col(df, ['date_block_num','shop_id'],'shop_cnt',{'item_cnt':'mean'})
# 当月某城市商品平均销量
df = agg_cnt_col(df, ['date_block_num','shop_city'],'city_cnt',{'item_cnt':'mean'})
#------------------------------


# %%
#------------------------------
# 新商品（item_age == 0)
def new_item_sales(df, merging_cols, new_col):
    temp = (
        df
        .query('item_age==0')
        .groupby(merging_cols)['item_cnt']
        .mean()
        .reset_index()
        .rename(columns={'item_cnt': new_col})
    )
    df = pd.merge(df, temp, on=merging_cols, how='left')
    return df

# mean units sold of new item in category at individual shop
df = new_item_sales(df, ['date_block_num','category_id','shop_id'], 'new_items_in_cat')
# mean units sold of new item in category across all shops
df = new_item_sales(df, ['date_block_num','category_id'], 'new_items_in_cat_all_shops')
#------------------------------


# %%
#------------------------------
def agg_price_col(df, merging_cols, new_col):
    temp = df.groupby(merging_cols).agg({'revenue':'sum','item_cnt_unclipped':'sum'}).reset_index()
    temp[new_col] = temp['revenue']/temp['item_cnt_unclipped']
    temp = temp[merging_cols + [new_col]]
    df = pd.merge(df, temp, on=merging_cols, how='left')
    return df

# 某商品的月平均价格
df = agg_price_col(df,['date_block_num','item_id'],'item_price')
# 某类别的月平均价格
df = agg_price_col(df,['date_block_num','category_id'],'category_price')
# 某月所有商品的平均价格
df = agg_price_col(df,['date_block_num'],'block_price')

df = downcast(df)
#------------------------------


## 1.6 添加滞后信息
# %%
#------------------------------
def lag_feature(df, lag, col, merge_cols):        
    temp = df[merge_cols + [col]]
    temp = temp.groupby(merge_cols).agg({f'{col}':'first'}).reset_index() # 每件商品的销量，因为item_id 唯一，因此 first 即代表总共销量
    temp.columns = merge_cols + [f'{col}_lag{lag}']
    temp['date_block_num'] += lag
    df = pd.merge(df, temp, on=merge_cols, how='left')
    df[f'{col}_lag{lag}'] = df[f'{col}_lag{lag}'].fillna(0).astype('float32')
    return df
#------------------------------

# %%
#------------------------------
# lag1 代表1月以前的数据；lag2 代表两个月以前的数据； lag1to12代表过去12个月的数据之和
# lag 12 的列
lag12_cols = {
    'item_cnt':['date_block_num', 'shop_id', 'item_id'],
    'item_cnt_all_shops':['date_block_num', 'item_id'],
    'category_cnt':['date_block_num', 'shop_id', 'category_id'],
    'category_cnt_all_shops':['date_block_num', 'category_id'],
    'group_cnt':['date_block_num', 'shop_id', 'group_id'],
    'group_cnt_all_shops':['date_block_num', 'group_id'],
    'shop_cnt':['date_block_num', 'shop_id'],
    'city_cnt':['date_block_num', 'shop_city'],
    'new_items_in_cat':['date_block_num', 'shop_id', 'category_id'],
    'new_items_in_cat_all_shops':['date_block_num', 'category_id']
}
for col,merge_cols in lag12_cols.items():
    df[f'{col}_lag1to12'] = 0
    for i in range(1,13):
        df = lag_feature(df, i, col, merge_cols)
        df[f'{col}_lag1to12'] += df[f'{col}_lag{i}']
        if i > 2:
            del df[f'{col}_lag{i}']
    if col == 'item_cnt':
        del df[f'{col}_lag1']
        del df[f'{col}_lag2']        
    else:
        del df[col] # 删除原始列，避免数据泄露
#------------------------------

# %%
#------------------------------
# 以下列，添加 lag1 和 lag2
lag2_cols = {
    'item_cnt_unclipped':['date_block_num', 'shop_id', 'item_id'],
    'item_cnt_all_shops_median':['date_block_num', 'item_id'],
    'category_cnt_median':['date_block_num', 'shop_id', 'category_id'],
    'category_cnt_all_shops_median':['date_block_num', 'category_id']
}
for col in lag2_cols:
    df = lag_feature(df, 1, col, merge_cols)
    df = lag_feature(df, 2, col, merge_cols)
    if col!='item_cnt_unclipped':
        del df[col]
#------------------------------

# %%
#------------------------------
# lag1 / lag1to12
df['item_cnt_diff'] = df['item_cnt_unclipped_lag1']/df['item_cnt_lag1to12']
df['item_cnt_all_shops_diff'] = df['item_cnt_all_shops_lag1']/df['item_cnt_all_shops_lag1to12']
df['category_cnt_diff'] = df['category_cnt_lag1']/df['category_cnt_lag1to12']
df['category_cnt_all_shops_diff'] = df['category_cnt_all_shops_lag1']/df['category_cnt_all_shops_lag1to12']
#------------------------------

# %%
#------------------------------
# 添加 lag1
df = lag_feature(df, 1, 'category_price',['date_block_num', 'category_id'])
df = lag_feature(df, 1, 'block_price',['date_block_num'])
del df['category_price'], df['block_price']
#------------------------------

# %%
#------------------------------
# 填充缺失值
df.loc[(df['item_age']>0) & (df['item_cnt_lag1to12'].isna()), 'item_cnt_lag1to12'] = 0
df.loc[(df['category_age']>0) & (df['category_cnt_lag1to12'].isna()), 'category_cnt_lag1to12'] = 0
df.loc[(df['group_age']>0) & (df['group_cnt_lag1to12'].isna()), 'group_cnt_lag1to12'] = 0
#------------------------------

# %%
#------------------------------
# df.query("item_cnt_lag1to12 > item_age")[['item_age', 'shop_age', 'item_cnt_lag1to12']]
# [min(idx) for idx in zip([11,2,34], [34, 34, 34], [12,12,12])]
# 由于有的商品的寿命不满12，因此不能算作是过去12个月的总销售量，需要除以月份，以用作过去n月的平均销量
df['item_cnt_lag1to12'] /= [min(idx) for idx in zip(df['item_age'],df['shop_age'],[12]*len(df))]
df['item_cnt_all_shops_lag1to12'] /= [min(idx) for idx in zip(df['item_age'],[12]*len(df))]
df['category_cnt_lag1to12'] /= [min(idx) for idx in zip(df['category_age'],df['shop_age'],[12]*len(df))]
df['category_cnt_all_shops_lag1to12'] /= [min(idx) for idx in zip(df['category_age'],[12]*len(df))]
df['group_cnt_lag1to12'] /= [min(idx) for idx in zip(df['group_age'],df['shop_age'],[12]*len(df))]
df['group_cnt_all_shops_lag1to12'] /= [min(idx) for idx in zip(df['group_age'],[12]*len(df))]
df['city_cnt_lag1to12'] /= [min(idx) for idx in zip(df['date_block_num'],[12]*len(df))]
df['shop_cnt_lag1to12'] /= [min(idx) for idx in zip(df['shop_age'],[12]*len(df))]
df['new_items_in_cat_lag1to12'] /= [min(idx) for idx in zip(df['category_age'],df['shop_age'],[12]*len(df))]
df['new_items_in_cat_all_shops_lag1to12'] /= [min(idx) for idx in zip(df['category_age'],[12]*len(df))]


df = downcast(df)
#------------------------------


# %%
#------------------------------
# 有些列需要用到较多的过去的信息，不适合只用lag 计算
def past_information(df, merging_cols, new_col, aggregation):
    temp = []
    for i in range(1,35):
        block = df.query(f'date_block_num < {i}').groupby(merging_cols).agg(aggregation).reset_index()
        block.columns = merging_cols + [new_col]
        block['date_block_num'] = i
        block = block[block[new_col]>0]
        temp.append(block)
    temp = pd.concat(temp)
    df = pd.merge(df, temp, on=['date_block_num']+merging_cols, how='left')
    return df

# 该item过去n月最后一次售出的平均价格 1 <= n <= 33
df = past_information(df, ['item_id'],'last_item_price',{'item_price':'last'})
# 该item过去n月在每个商店的销量
df = past_information(df, ['shop_id','item_id'],'item_cnt_sum_alltime',{'item_cnt':'sum'})
# 该item过去n月的总销量
df = past_information(df, ['item_id'],'item_cnt_sum_alltime_allshops',{'item_cnt':'sum'})

# 下面的列不再需要，因为会导致数据泄露问题（data leakage)
del df['revenue'], df['item_cnt_unclipped'], df['item_price']
#------------------------------


# %%
#------------------------------
# 某item 上月最后一次售出的平均价格 / 上月所有商品的均价
df['relative_price_item_block_lag1'] = df['last_item_price']/df['block_price_lag1']
# 自从某件商品第一次售出后，平均日销售价格
df['item_cnt_per_day_alltime'] = (df['item_cnt_sum_alltime']/df['prev_days_on_sale']).fillna(0)
df['item_cnt_per_day_alltime_allshops'] = (df['item_cnt_sum_alltime_allshops']/df['prev_days_on_sale']).fillna(0)

import gc
gc.collect()
df = downcast(df)
#------------------------------


# %%
#------------------------------
def matching_name_cat_age(df,n,all_shops):
    temp_cols = [f'same_name{n}catage_cnt','date_block_num', f'item_name_first{n}','item_age','category_id']
    if all_shops:
        temp_cols[0] += '_all_shops'
    else:
        temp_cols += ['shop_id']
    temp = []
    for i in range(1,35):
        block = (
            df
            .query(f'date_block_num < {i}')
            .groupby(temp_cols[2:])
            .agg({'item_cnt':'mean'})
            .reset_index()
            .rename(columns={'item_cnt':temp_cols[0]})
        )
        block = block[block[temp_cols[0]]>0]
        block['date_block_num'] = i
        temp.append(block)
    temp = pd.concat(temp)
    df = pd.merge(df, temp, on=temp_cols[1:], how='left')
    return df

for n in [4,6,11]:
    for all_shops in [True,False]:
        df = matching_name_cat_age(df,n,all_shops)
#------------------------------


# %%
#------------------------------
#assign appropriate datatypes
df = downcast(df)
int8_cols = [
    'item_cnt','month','group_id','shop_type',
    'shop_city','shop_id','date_block_num','category_id',
    'item_age',
]
int16_cols = [
    'item_id','item_name_first4',
    'item_name_first6','item_name_first11'
]
for col in int8_cols:
    df[col] = df[col].astype('int8')
for col in int16_cols:
    df[col] = df[col].astype('int16')
#------------------------------



# %%
#------------------------------
# 添加item_name_firstn 的过去信息
def matching_name_cat_age(df,n,all_shops):
    temp_cols = [f'same_name{n}catage_cnt','date_block_num', f'item_name_first{n}','item_age','category_id']
    if all_shops:
        temp_cols[0] += '_all_shops'
    else:
        temp_cols += ['shop_id']
    temp = []
    for i in range(1,35):
        block = (
            df
            .query(f'date_block_num < {i}')
            .groupby(temp_cols[2:])
            .agg({'item_cnt':'mean'})
            .reset_index()
            .rename(columns={'item_cnt':temp_cols[0]})
        )
        block = block[block[temp_cols[0]]>0]
        block['date_block_num'] = i
        temp.append(block)
    temp = pd.concat(temp)
    df = pd.merge(df, temp, on=temp_cols[1:], how='left')
    return df

for n in [4,6,11]:
    for all_shops in [True,False]:
        df = matching_name_cat_age(df,n,all_shops)
#------------------------------


# %%
#------------------------------
def nearby_item_data(df,col):
    if col in ['item_cnt_unclipped_lag1','item_cnt_lag1to12']:
        cols = ['date_block_num', 'shop_id', 'item_id']
        temp = df[cols + [col]] 
    else:
        cols = ['date_block_num', 'item_id']
        temp = df.groupby(cols).agg({col:'first'}).reset_index()[cols + [col]]   
    
    temp.columns = cols + [f'below_{col}']
    temp['item_id'] += 1
    df = pd.merge(df, temp, on=cols, how='left')
    
    temp.columns = cols + [f'above_{col}']
    temp['item_id'] -= 2
    df = pd.merge(df, temp, on=cols, how='left')
    
    return df

item_cols = ['item_cnt_unclipped_lag1','item_cnt_lag1to12',
             'item_cnt_all_shops_lag1','item_cnt_all_shops_lag1to12']
for col in item_cols:
    df = nearby_item_data(df,col)
    
del temp
#------------------------------


## 1.7 Encoding name information
# %%
#------------------------------
# 添加0/1变量——item_name 是否包含常见的word
results = Counter()
items['item_name'].str.split().apply(results.update)

words = []
cnts = []
for key, value in results.items():
    words.append(key)
    cnts.append(value)
    
counts = pd.DataFrame({'word':words,'count':cnts})
common_words = counts.query('count>200').word.to_list()
for word in common_words:
    items[f'{word}_in_name'] = items['item_name'].str.contains(word).astype('int8')
drop_cols = [
    'item_id','category_id','item_name','item_name_first4',
    'item_name_first6','item_name_first11',
    'category_name','group_name','group_id'
]
items = items.drop(columns=drop_cols)

df = df.join(items, on='item_id')
#------------------------------