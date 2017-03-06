# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 16:23:38 2017

@author: 呵呵
"""
#pandas常用命令
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series,DataFrame
#------------------------------------------------------------------------------
#一、创建对象
#------------------------------------------------------------------------------

#1、可以通过传递一个list对象来创建一个Series，pandas会默认创建整型索引：
s = pd.Series([1,3,5,np.nan,6,8])

#2、通过传递一个numpy array，时间索引以及列标签来创建一个DataFrame：

dates = pd.date_range('20130101', periods=6)
#df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))

#3、通过传递一个能够被转换成类似序列结构的字典对象来创建一个DataFrame：
df2 = pd.DataFrame({ 'A' : 1.,
                     'B' : pd.Timestamp('20130102'),
                     'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                     'D' : np.array([23] * 4,dtype='int32'),
                     'E' : pd.Categorical(["test","train","test","train"]),
                     'F' : 'foo' })


df3 = { 'A' : 1.,
                     'B' : pd.Timestamp('20130102'),
                     'C' : pd.Series(1,index=list(range(10)),dtype ='float32'),
                     'D' : np.array([23] * 4,dtype='int32'),
                     'E' : pd.Categorical(["test","train","test","train"]),
                     'F' : 'foo' }
                     
df4 = pd.DataFrame({"A": np.array([23]),
                    "hahah" : pd.Series(1,range(10))
                    },index =pd.date_range("20170109",periods = 10 ))                     
                     
print(df2)
print(df3)
#4、查看不同列的数据类型
df2.dtypes

#------------------------------------------------------------------------------
#二、查看数据
#------------------------------------------------------------------------------
#1、查看frame中头部和尾部的行：
df.head()
df.tail()

#2、显示索引、列和底层的numpy数据：
df.index
df.columns
df.values

#3、describe()函数对于数据的快速统计汇总：
df.describe()

#4、对数据的转置
df.T

#5、按轴进行排序 由小到大
df.sort_index(axis=1, ascending=False) #ascending = True :由小到大

#6、按值进行排序
df.sort_values(by='B')


#------------------------------------------------------------------------------
#三、选择
#------------------------------------------------------------------------------


#获取
#-----------------------------------------------------------------------------


#1、 选择一个单独的列，这将会返回一个Series，等同于df.A：
df["A"]

#2、 通过[]进行选择，这将会对行进行切片
df[0:3]
df["20130102":"20130104"]


#通过标签选择
#-----------------------------------------------------------------------------

#1、使用标签来获取一个交叉的区域
df.loc[dates[0]]

#2、 通过标签来在多个轴上进行选择
df.loc[:,['A','B']]

#3、 标签切片
df.loc['20130102':'20130104',['A','B']]
df.loc[dates[0:3],['A','B']]

#4、 对于返回的对象进行维度缩减
df.loc['20130102',['A','B']]


#5、 获取一个标量
df.loc[dates[0],'A']


#6、 快速访问一个标量（与上一个方法等价）
df.at[dates[0],'A']


#通过位置选择
#-----------------------------------------------------------------------------

#1、 通过传递数值进行位置选择（选择的是行）
df.iloc[3]


#2、 通过数值进行切片，与numpy/python中的情况类似
df.iloc[3:5,0:2]


#3、 通过指定一个位置的列表，与numpy/python中的情况类似
df.iloc[[1,2,4],[0,2]]


#4、 对行进行切片
df.iloc[1:3,:]


#5、 对列进行切片
df.iloc[:,1:3]


#6、 获取特定的值
df.iloc[1,1]
df.iat[1,1]



#布尔索引
#-----------------------------------------------------------------------------

#1、 使用一个单独列的值来选择数据：
df[df.A > 0]


#2、 使用where操作来选择数据：
df[df > 0]


#3、 使用isin()方法来过滤：
In [41]: df2 = df.copy()

In [42]: df2['E'] = ['one', 'one','two','three','four','three']

In [43]: df2

In [44]: df2[df2['E'].isin(['two','four'])]


#设置
#-----------------------------------------------------------------------------

#1、 设置一个新的列：

In [45]: s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))

In [46]: s1

In [47]: df['F'] = s1


#2、 通过标签设置新的值：

df.at[dates[0],'A'] = 0

#3、 通过位置设置新的值：

df.iat[0,1] = 0

#4、 通过一个numpy数组设置一组新值：

df.loc[:,'D'] = np.array([5] * len(df))
df.loc[:,'A'] = np.array(np.random.randn(len(df),1))
np.array(range(len(df)))


#5、 通过where操作来设置新的值：把数据中正值全部转为负，负的保留不变

In [52]: df2 = df.copy()

In [53]: df2[df2 > 0] = -df2

In [54]: df2



#------------------------------------------------------------------------------
#四、缺失值处理
#-----------------------------------------------------------------------------
#在pandas中，使用np.nan来代替缺失值，这些值将默认不会包含在计算中.

#1、reindex()方法可以对指定轴上的索引进行改变/增加/删除操作，这将返回原始数据的一个拷贝：
In [55]: df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])

In [56]: df1.loc[dates[0]:dates[1],'E'] = 1

In [57]: df1


#2、  去掉包含缺失值的行：
In [58]: df1.dropna(how='any')


#3、  对缺失值进行填充：
In [59]: df1.fillna(value=5)


#4、  对数据进行布尔填充：
In [60]: pd.isnull(df1)


#-----------------------------------------------------------------------------
#五、相关操作
#-----------------------------------------------------------------------------



#统计
#-----------------------------------------------------------------------------
#1、  执行描述性统计：

df.mean()

#2、  在其他轴上进行相同的操作：

df.mean(1)

#3、  对于拥有不同维度，需要对齐的对象进行操作。Pandas会自动的沿着指定的维度进行广播：
#没搞懂是看啥的

In [63]: s = pd.Series([1,3,5,np.nan,6,8], index=dates).shift(2)

In [64]: s

df.sub(s, axis='index')


#Apply
#-----------------------------------------------------------------------------
#将函数应用到数据中

df.apply(np.cumsum)


df.apply(lambda x: x.max() - x.min())




#直方图
#-----------------------------------------------------------------------------

np.random.randint(0, 7, size=10)) # 指定最大最小值及数量获取一个整数list
In [68]: s = pd.Series(np.random.randint(0, 7, size=10))

In [69]: s

In [70]: s.value_counts() #计算值的数量





#字符串方法
#-----------------------------------------------------------------------------
#Series对象在其str属性中配备了一组字符串处理方法，可以很容易的应用到数组中的每个元素
#，如下段代码所示。更多详情请参考：Vectorized String Methods.

In [71]: s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])

In [72]: s.str.lower()#把所有字母小写



#-----------------------------------------------------------------------------
#六、合并
#-----------------------------------------------------------------------------
#Pandas提供了大量的方法能够轻松的对Series，DataFrame和Panel对象进行各种符合各种逻辑
#关系的合并操作。具体请参阅：Merging section


#Concat()
#-----------------------------------------------------------------------------


In [73]: df = pd.DataFrame(np.random.randn(10, 4))

In [74]: df

In [75]: pieces = [df[:3], df[3:7], df[7:]]

In [76]: pd.concat(pieces)



#Join
#-----------------------------------------------------------------------------
#类似于SQL类型的合并，具体请参阅

In [77]: left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})

In [78]: right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})

In [79]: left
In [80]: right
In [81]: pd.merge(left, right, on='key')



#另外一个例子：
#-----------------------------------------------------------------------------
In [82]: left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})

In [83]: right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})

In [84]: left

In [85]: right

In [86]: pd.merge(left, right, on='key')



#Append()
#-----------------------------------------------------------------------------
将一行连接到一个DataFrame上，

In [87]: df = pd.DataFrame(np.random.randn(8, 4), columns=['A','B','C','D'])

In [88]: df

In [89]: s = df.iloc[3]

In [90]: df.append(s, ignore_index=True) # 忽略索引仅添加数据


#-----------------------------------------------------------------------------
#七、分组
#-----------------------------------------------------------------------------

#对于”group by”操作，我们通常是指以下一个或多个操作步骤：
#（Splitting）按照一些规则将数据分为不同的组；
#（Applying）对于每组数据分别执行一个函数；
#（Combining）将结果组合到一个数据结构中；


In [91]: df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
   ....:                           'foo', 'bar', 'foo', 'foo'],
   ....:                    'B' : ['one', 'one', 'two', 'three',
   ....:                           'two', 'two', 'one', 'three'],
   ....:                    'C' : np.random.randn(8),
   ....:                    'D' : np.random.randn(8)})
   ....: 


#1、  分组并对每个分组执行sum函数：

In [93]: df.groupby('A').sum()

#2、  通过多个列进行分组形成一个层次索引，然后执行函数：

In [94]: df.groupby(['A','B']).sum()





#-----------------------------------------------------------------------------
#八、重塑
#-----------------------------------------------------------------------------


#Stack
#-----------------------------------------------------------------------------
In [95]: tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
   ....:                      'foo', 'foo', 'qux', 'qux'],
   ....:                     ['one', 'two', 'one', 'two',
   ....:                      'one', 'two', 'one', 'two']]))
   ....: 

In [96]: index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])

#MultiIndex多重索引的创建

In [97]: df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])

In [98]: df2 = df[:4]

In [99]: df2


#The stack() method “compresses” a level in the DataFrame’s columns.

In [100]: stacked = df2.stack()

"""
first  second   
bar    one     A   -0.711239
               B    3.348976
       two     A    0.759271
               B    0.342174
baz    one     A   -0.785090
               B   -0.027236
       two     A   -0.071495
               B    0.184496
foo    one     A   -0.020221
               B    0.792804
       two     A   -0.988705
               B   -1.371141
qux    one     A    0.847719
               B    0.054975
       two     A   -2.123057
               B    0.475634
"""



In [102]: stacked.unstack()

In [103]: stacked.unstack(1)

In [104]: stacked.unstack(0)



#数据透视表
#-----------------------------------------------------------------------------



In [105]: df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
   .....:                    'B' : ['A', 'B', 'C'] * 4,
   .....:                    'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
   .....:                    'D' : np.random.randn(12),
   .....:                    'E' : np.random.randn(12)})
   .....: 


#可以从这个数据中轻松的生成数据透视表：

In [107]: pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])
"""
Out[107]: 
C             bar       foo
A     B                    
one   A -0.773723  1.418757
      B -0.029716 -1.879024
      C -1.146178  0.314665
three A  1.006160       NaN
      B       NaN -1.035018
      C  0.648740       NaN
two   A       NaN  0.100900
      B -1.170653       NaN
      C       NaN  0.536826

"""


#-----------------------------------------------------------------------------
#九、时间序列
#-----------------------------------------------------------------------------
#Pandas在对频率转换进行重新采样时拥有简单、强大且高效的功能（如将按秒采样的数据转换
#为按5分钟为单位进行采样的数据）。这种操作在金融领域非常常见。


In [108]: rng = pd.date_range('1/1/2012', periods=100, freq='S')

In [109]: ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)

In [110]: ts.resample('5Min').sum()



#1、时区表示：
#-----------------------------------------------------------------------------

In [111]: rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')

In [112]: ts = pd.Series(np.random.randn(len(rng)), index = rng)

In [113]: ts

In [114]: ts_utc = ts.tz_localize('UTC')

In [115]: ts_utc



#2、时区转换：
#-----------------------------------------------------------------------------
In [116]: ts_utc.tz_convert('US/Eastern')




#3、 时间跨度转换：
#-----------------------------------------------------------------------------
In [117]: rng = pd.date_range('1/1/2012', periods=5, freq='M')

In [118]: ts = pd.Series(np.random.randn(len(rng)), index=rng)

In [119]: ts

In [120]: ps = ts.to_period()

In [121]: ps

In [122]: ps.to_timestamp()



#4、时期和时间戳之间的转换使得可以使用一些方便的算术函数。
#-----------------------------------------------------------------------------
In [123]: prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')

In [124]: ts = pd.Series(np.random.randn(len(prng)), prng)

In [125]: ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9
#prng.asfreq('M', 'e') 其中第二项分别代表Start 和End 代表时间末
In [126]: ts.head()


#-----------------------------------------------------------------------------
#十、Categorical
#-----------------------------------------------------------------------------
#从0.15版本开始，pandas可以在DataFrame中支持Categorical类型的数据，详细 介绍参看

df = pd.DataFrame({"id":[1,2,3,4,5,6], "raw_grade":['a', 'b', 'b', 'a', 'a', 'e']})

#1、  将原始的grade转换为Categorical数据类型：
In [128]: df["grade"] = df["raw_grade"].astype("category")

In [129]: df["grade"]


#2、  将Categorical类型数据重命名为更有意义的名称：
In [130]: df["grade"].cat.categories = ["very good", "good", "very bad"]


#3、  对类别进行重新排序，增加缺失的类别：

In [131]: df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])

In [132]: df["grade"]

#4、  排序是按照Categorical的顺序进行的而不是按照字典顺序进行：

In [133]: df.sort_values(by="grade")

#5、  对Categorical列进行排序时存在空的类别：

In [134]: df.groupby("grade").size()




#-----------------------------------------------------------------------------
#十一、画图
#-----------------------------------------------------------------------------

In [135]: ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))

In [136]: ts = ts.cumsum()

In [137]: ts.plot()


#对于DataFrame来说，plot是一种将所有列及其标签进行绘制的简便方法：

In [138]: df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,
   .....:                   columns=['A', 'B', 'C', 'D'])
   .....: 

In [139]: df = df.cumsum()

In [140]: plt.figure(); df.plot(); plt.legend(loc='best')







#-----------------------------------------------------------------------------
#十二、导入和保存数据
#-----------------------------------------------------------------------------


#1、写入csv文件：
#-----------------------------------------------------------------------------
df.to_csv('foo.csv')

#2、  从csv文件中读取：

pd.read_csv('foo.csv')




#2、写入HDF5存储：
#-----------------------------------------------------------------------------
df.to_hdf('foo.h5','df')

#2、从HDF5存储中读取：

pd.read_hdf('foo.h5','df')





#3、写入excel文件：
#-----------------------------------------------------------------------------
df.to_excel('foo.xlsx', sheet_name='Sheet1')

#2、从excel文件中读取：

pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])





#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------




#---------------------------《利用python进行数据分析》--------------------------

#Series

#-----------------------------------------------------------------------------
#创建一个Series
obj = Series([4,5,1,2])

obj.values 
Out[11]: array([4, 5, 1, 2], dtype=int64)

obj.index
RangeIndex(start=0, stop=4, step=1)


#创建一个字母索引的Series
obj2 = Series([1,2,3,4,1],index  = ['a','b','c','d','e'])

obj2['a']
Out[18]: 1

obj2[['a','b','d']]
Out[20]: 
a    1
b    2
d    4
dtype: int64

#-----------------------------------------------------------------------------
obj2[obj2>2]        #选取obj》2的部分并在obj中显示
Out[26]: 
c    3
d    4
dtype: int64

obj2*2
np.exp(obj2)        #计算e的次方
pd.isnull(obj)      #检测缺失数据
pd.isnotnull(obj)   #检测缺失数据
obj.isnull()        #检测缺失数据
obj.name = 'XXXX'  #给Series的值设置一个名称
obj.index.name = 'XXXX'  #给Series的索引设置一个名称

obj.index = ['a','b','c','d'] #修改Series的索引


#-----------------------------------------------------------------------------

#DataFram


data = {'statw':['owio','owio','owio','haha','haha'],
        'year' :[2000,2001,2002,2003,2004],
        'pop'  :[1.5,1.7,1.3,1.3,1.9]
        }

data = pd.DataFrame(data)

DataFrame(data,columns = ['year','statw','pop']) #设置列序列
DataFrame(data,columns = ['year','hehehe','statw','pop','hahaha'])  # 新增两个列序列
data.columns










