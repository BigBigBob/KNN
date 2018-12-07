

```python
import numpy as np
import operator
```

### 定义一个分类器
inX —— 用于分类的数据（即测试集）  
dataSet —— 训练集   
labels —— 训练集对应的分类标签  
k —— 选择前k个点


```python
def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]  #获得第一维的长度，即行数，即点的数量
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet #首先以inx为基底构建 dataSetSize行,1列的数组；然后与训练集做差
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1) #对每一行进行求和运算
    distances=sqDistances**0.5
    sortedDistIndicies=distances.argsort() #返回的是数组值从小到大的索引值
    classCount={} #字典
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]] #获得距离前k的标签
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1 #对标签进行计数
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True) #从大到小进行排序
    return sortedClassCount[0][0] #返回次数最多的类别作为结果
```

注： python3中用items()替换python2中的iteritems()  
否则报如下的错：  
AttributeError: 'dict' object has no attribute 'iteritems'

### 从文件中获取数据
filename —— 文件名


```python
def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()  #按行读取文件
    numberOfLines=len(arrayOLines)  #获得文件的行数
    returnMat=np.zeros((numberOfLines,3))  #生成0矩阵 行数为文件的行数，列数为3
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()
        listFromLine=line.split('\t')  #将字符串根据'\t'分隔符进行切片
        returnMat[index,:]=listFromLine[0:3]  #提取前三列数据作为特征矩阵
        classLabelVector.append(int(listFromLine[-1])) #提取最后一列作为分类标签
        index+=1
    return returnMat,classLabelVector  #返回特征矩阵与分类标签
```

### 对数据进行归一化处理
dataSet —— 特征矩阵


```python
def autoNorm(dataSet):
    maxVals=dataSet.max(0) #按列选择最大值
    minVals=dataSet.min(0) #按列选择最小值
    
    ranges=maxVals-minVals 
    normDataSet=np.zeros(np.shape(dataSet))  
    m=dataSet.shape[0]
    normDataSet=dataSet-np.tile(minVals,(m,1))  #原始值减去最小值
    normDataSet=normDataSet/np.tile(ranges,(m,1)) #原始值减去最小值除以最大值与最小值的差
    return normDataSet,ranges,minVals  #返回 归一化后的特征矩阵，数据范围，数据最小值
```

### 输入输出类


```python
def classifyPerson():
    resultList=['not at all','in small doses','in large doses']
    percentTats=float(input("percentage of time spent palying video games?"))
    ffMiles=float(input("frequent filer miles earned per year?"))
    iceCream=float(input("liters of iceCream consumed per year?"))
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat) #归一化处理
    inArr=np.array([percentTats,ffMiles,iceCream])
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("you will probably like this person:",resultList[classifierResult-1])
```


```python
classifyPerson()
```

    percentage of time spent palying video games?342
    frequent filer miles earned per year?234
    liters of iceCream consumed per year?123
    you will probably like this person: in large doses

