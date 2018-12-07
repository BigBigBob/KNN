

```python
import numpy as np
import operator
from os import listdir
```

### 将文件转换为向量
将32\*32 转化为 1\*1024


```python
def img2vector(filename):
    returnVect=np.zeros((1,1024)) #构造一个1行1024列的零矩阵
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()  #按行读取文件
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j]) #将每行的数据按列替换零矩阵中的数据
    return returnVect #返回转化后的1行1024列的矩阵
```

### 分类器


```python
def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0] #获得第一维的长度，即行数，即点的数量
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet #首先以inx为基底构建 dataSetSize行,1列的数组；然后与训练集做差
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1) #按行
    distances=sqDistances**0.5
    sortedDistIndicies=distances.argsort() #返回的是数组值从小到大的索引值
    classCount={} #字典
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]] #获得距离前k的标签
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1 #对标签进行计数
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True) #从大到小进行排序
    return sortedClassCount[0][0]

```

### 测试算法


```python
def handwritingClassTest():
    hwLabels=[]
    trainingFileList=listdir('trainingDigits') #获得trainingDigits目录下的所有文件
    m=len(trainingFileList)  #获得文件的数量，每个文件中的内容为32*32的矩阵
    trainingMat=np.zeros((m,1024))  #构建一个m行，1024列的零矩阵。未来将每个文件转化为1行1024列的数据，存入trainingMat中作为训练集
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0]) #从文件名中获取标签，机智
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2vector('trainingDigits/%s' % fileNameStr) #传入文件名，将这个文件内的32*32矩阵转换为1*1024
    testFileList=listdir('testDigits') #获得testDigits目录下的所有文件--作为测试集
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=img2vector('testDigits/%s' % fileNameStr)
        classifyResult=classify0(vectorUnderTest,trainingMat,hwLabels,3) #传入分类器中分类
        print("the classifier came back with:%d,the real answer is %d" % (classifyResult,classNumStr))
        if(classifyResult!=classNumStr):
            errorCount+=1.0
        print("the total number of error is %d" % errorCount)
        print("total error rate is: %f " % (errorCount/float(mTest)))
```


```python
handwritingClassTest()
```

    the classifier came back with:2,the real answer is 2
    the total number of error is 0
    total error rate is: 0.000000 
    the classifier came back with:9,the real answer is 9
    the total number of error is 0
    total error rate is: 0.000000 
    the classifier came back with:4,the real answer is 4
    the total number of error is 0
    total error rate is: 0.000000 
    the classifier came back with:3,the real answer is 3
    the total number of error is 0
    total error rate is: 0.000000 
    the classifier came back with:8,the real answer is 8
    the total number of error is 0
    total error rate is: 0.000000 
    the classifier came back with:4,the real answer is 4
    the total number of error is 0
    total error rate is: 0.000000 
    the classifier came back with:6,the real answer is 6
    the total number of error is 0
    total error rate is: 0.000000 
    the classifier came back with:1,the real answer is 1
    the total number of error is 0
    total error rate is: 0.000000 
    the classifier came back with:7,the real answer is 7
    the total number of error is 0
    total error rate is: 0.000000 
    the classifier came back with:1,the real answer is 1
    the total number of error is 0
    total error rate is: 0.000000 
    the classifier came back with:3,the real answer is 5
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:5,the real answer is 5
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:3,the real answer is 3
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:2,the real answer is 2
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:6,the real answer is 6
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:9,the real answer is 9
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:0,the real answer is 0
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:9,the real answer is 9
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:6,the real answer is 6
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:0,the real answer is 0
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:6,the real answer is 6
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:5,the real answer is 5
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:1,the real answer is 1
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:8,the real answer is 8
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:8,the real answer is 8
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:4,the real answer is 4
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:6,the real answer is 6
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:2,the real answer is 2
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:5,the real answer is 5
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:1,the real answer is 1
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:6,the real answer is 6
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:4,the real answer is 4
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:1,the real answer is 1
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:3,the real answer is 3
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:4,the real answer is 4
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:1,the real answer is 1
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:8,the real answer is 8
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:6,the real answer is 6
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:9,the real answer is 9
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:5,the real answer is 5
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:1,the real answer is 1
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:3,the real answer is 3
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:1,the real answer is 1
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:1,the real answer is 1
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:7,the real answer is 7
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:5,the real answer is 5
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:6,the real answer is 6
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:1,the real answer is 1
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:4,the real answer is 4
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:7,the real answer is 7
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:8,the real answer is 8
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:0,the real answer is 0
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:5,the real answer is 5
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:5,the real answer is 5
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:4,the real answer is 4
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:0,the real answer is 0
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:0,the real answer is 0
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:1,the real answer is 1
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:6,the real answer is 6
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:7,the real answer is 7
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:7,the real answer is 7
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:4,the real answer is 4
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:6,the real answer is 6
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:9,the real answer is 9
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:0,the real answer is 0
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:7,the real answer is 7
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:1,the real answer is 1
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:2,the real answer is 2
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:8,the real answer is 8
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:5,the real answer is 5
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:0,the real answer is 0
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:7,the real answer is 7
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:7,the real answer is 7
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:5,the real answer is 5
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:1,the real answer is 1
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:6,the real answer is 6
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:1,the real answer is 1
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:7,the real answer is 7
    the total number of error is 1
    total error rate is: 0.001057 
    the classifier came back with:1,the real answer is 9
    the total number of error is 2
    total error rate is: 0.002114 
    the classifier came back with:0,the real answer is 0
    the total number of error is 2
    total error rate is: 0.002114 
    the classifier came back with:1,the real answer is 1
    the total number of error is 2
    total error rate is: 0.002114 
    the classifier came back with:3,the real answer is 3
    the total number of error is 2
    total error rate is: 0.002114 
    the classifier came back with:3,the real answer is 3
    the total number of error is 2
    total error rate is: 0.002114 
    the classifier came back with:7,the real answer is 7
    the total number of error is 2
    total error rate is: 0.002114 
    the classifier came back with:4,the real answer is 4
    the total number of error is 2
    total error rate is: 0.002114 
    the classifier came back with:8,the real answer is 8
    the total number of error is 2
    total error rate is: 0.002114 
    the classifier came back with:6,the real answer is 5
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:4,the real answer is 4
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:2,the real answer is 2
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:3,the real answer is 3
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:2,the real answer is 2
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:4,the real answer is 4
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:5,the real answer is 5
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:5,the real answer is 5
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:7,the real answer is 7
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:7,the real answer is 7
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:9,the real answer is 9
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:0,the real answer is 0
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:0,the real answer is 0
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:7,the real answer is 7
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:1,the real answer is 1
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:2,the real answer is 2
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:9,the real answer is 9
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:8,the real answer is 8
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:6,the real answer is 6
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:1,the real answer is 1
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:5,the real answer is 5
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:4,the real answer is 4
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:7,the real answer is 7
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:5,the real answer is 5
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:0,the real answer is 0
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:2,the real answer is 2
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:1,the real answer is 1
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:5,the real answer is 5
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:0,the real answer is 0
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:6,the real answer is 6
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:0,the real answer is 0
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:6,the real answer is 6
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:4,the real answer is 4
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:9,the real answer is 9
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:5,the real answer is 5
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:8,the real answer is 8
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:9,the real answer is 9
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:7,the real answer is 7
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:6,the real answer is 6
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:5,the real answer is 5
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:7,the real answer is 7
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:0,the real answer is 0
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:9,the real answer is 9
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:7,the real answer is 7
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:4,the real answer is 4
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:6,the real answer is 6
    the total number of error is 3
    total error rate is: 0.003171 
    the classifier came back with:5,the real answer is 9
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:4,the real answer is 4
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:4,the real answer is 4
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:4,the real answer is 4
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:4,the real answer is 4
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:4,the real answer is 4
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:4,the real answer is 4
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:9,the real answer is 9
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:0,the real answer is 0
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:6,the real answer is 6
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:1,the real answer is 1
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:6,the real answer is 6
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:3,the real answer is 3
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:3,the real answer is 3
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:0,the real answer is 0
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:2,the real answer is 2
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:2,the real answer is 2
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:8,the real answer is 8
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:4,the real answer is 4
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:9,the real answer is 9
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:1,the real answer is 1
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:1,the real answer is 1
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:2,the real answer is 2
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:8,the real answer is 8
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:1,the real answer is 1
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:2,the real answer is 2
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:7,the real answer is 7
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:6,the real answer is 6
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:6,the real answer is 6
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:8,the real answer is 8
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:6,the real answer is 6
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:4,the real answer is 4
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:6,the real answer is 6
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:9,the real answer is 9
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:3,the real answer is 3
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:0,the real answer is 0
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:8,the real answer is 8
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:6,the real answer is 6
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:3,the real answer is 3
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:8,the real answer is 8
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:4,the real answer is 4
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:9,the real answer is 9
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:8,the real answer is 8
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:8,the real answer is 8
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:2,the real answer is 2
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:2,the real answer is 2
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:2,the real answer is 2
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:2,the real answer is 2
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:9,the real answer is 9
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:0,the real answer is 0
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:8,the real answer is 8
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:4,the real answer is 4
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:1,the real answer is 1
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:7,the real answer is 7
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:8,the real answer is 8
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:1,the real answer is 1
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:7,the real answer is 7
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:0,the real answer is 0
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:1,the real answer is 1
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:7,the real answer is 7
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:7,the real answer is 7
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:0,the real answer is 0
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:4,the real answer is 4
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:1,the real answer is 1
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:8,the real answer is 8
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:6,the real answer is 6
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:9,the real answer is 9
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:0,the real answer is 0
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:4,the real answer is 4
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:3,the real answer is 3
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:3,the real answer is 3
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:7,the real answer is 7
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:0,the real answer is 0
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:1,the real answer is 1
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:0,the real answer is 0
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:4,the real answer is 4
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:9,the real answer is 9
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:8,the real answer is 8
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:1,the real answer is 1
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:9,the real answer is 9
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:8,the real answer is 8
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:4,the real answer is 4
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:8,the real answer is 8
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:7,the real answer is 7
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:1,the real answer is 1
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:3,the real answer is 3
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:1,the real answer is 1
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:8,the real answer is 8
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:6,the real answer is 6
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:1,the real answer is 1
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:3,the real answer is 3
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:8,the real answer is 8
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:0,the real answer is 0
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:8,the real answer is 8
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:4,the real answer is 4
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:2,the real answer is 2
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:6,the real answer is 6
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:6,the real answer is 6
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:4,the real answer is 4
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:9,the real answer is 9
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:3,the real answer is 3
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:6,the real answer is 6
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:2,the real answer is 2
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:0,the real answer is 0
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:1,the real answer is 1
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:1,the real answer is 1
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:8,the real answer is 8
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:3,the real answer is 3
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:3,the real answer is 3
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:2,the real answer is 2
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:2,the real answer is 2
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:6,the real answer is 6
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:2,the real answer is 2
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:3,the real answer is 3
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:9,the real answer is 9
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:0,the real answer is 0
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:1,the real answer is 1
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:3,the real answer is 3
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:6,the real answer is 6
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:4,the real answer is 4
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:2,the real answer is 2
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:9,the real answer is 9
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:2,the real answer is 2
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:9,the real answer is 9
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:1,the real answer is 1
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:6,the real answer is 6
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:7,the real answer is 7
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:8,the real answer is 8
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:6,the real answer is 6
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:6,the real answer is 6
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:7,the real answer is 7
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:4,the real answer is 4
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:8,the real answer is 8
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:2,the real answer is 2
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:0,the real answer is 0
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:3,the real answer is 3
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:0,the real answer is 0
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:1,the real answer is 1
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:9,the real answer is 9
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:8,the real answer is 8
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:4,the real answer is 4
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:0,the real answer is 0
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:4,the real answer is 4
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:9,the real answer is 9
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:7,the real answer is 7
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:2,the real answer is 2
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:0,the real answer is 0
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:2,the real answer is 2
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:7,the real answer is 7
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:2,the real answer is 2
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:2,the real answer is 2
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:4,the real answer is 4
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:6,the real answer is 6
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:4,the real answer is 4
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:9,the real answer is 9
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:2,the real answer is 2
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:0,the real answer is 0
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:9,the real answer is 9
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:7,the real answer is 7
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:3,the real answer is 3
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:3,the real answer is 3
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:4,the real answer is 4
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:8,the real answer is 8
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:1,the real answer is 1
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:0,the real answer is 0
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:9,the real answer is 9
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:0,the real answer is 0
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:5,the real answer is 5
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:0,the real answer is 0
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:1,the real answer is 1
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:7,the real answer is 7
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:7,the real answer is 7
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:2,the real answer is 2
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:1,the real answer is 1
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:1,the real answer is 1
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:9,the real answer is 9
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:7,the real answer is 7
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:2,the real answer is 2
    the total number of error is 4
    total error rate is: 0.004228 
    the classifier came back with:3,the real answer is 8
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:5,the real answer is 5
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:4,the real answer is 4
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:3,the real answer is 3
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:6,the real answer is 6
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:6,the real answer is 6
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:5,the real answer is 5
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:0,the real answer is 0
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:7,the real answer is 7
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:6,the real answer is 6
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:1,the real answer is 1
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:7,the real answer is 7
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:4,the real answer is 4
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:2,the real answer is 2
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:2,the real answer is 2
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:3,the real answer is 3
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:4,the real answer is 4
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:9,the real answer is 9
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:7,the real answer is 7
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:0,the real answer is 0
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:9,the real answer is 9
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:0,the real answer is 0
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:6,the real answer is 6
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:2,the real answer is 2
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:1,the real answer is 1
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:5,the real answer is 5
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:0,the real answer is 0
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:7,the real answer is 7
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:4,the real answer is 4
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:4,the real answer is 4
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:5,the real answer is 5
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:9,the real answer is 9
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:5,the real answer is 5
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:7,the real answer is 7
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:2,the real answer is 2
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:0,the real answer is 0
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:6,the real answer is 6
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:5,the real answer is 5
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:5,the real answer is 5
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:3,the real answer is 3
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:2,the real answer is 2
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:2,the real answer is 2
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:7,the real answer is 7
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:7,the real answer is 7
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:6,the real answer is 6
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:0,the real answer is 0
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:5,the real answer is 5
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:4,the real answer is 4
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:4,the real answer is 4
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:6,the real answer is 6
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:8,the real answer is 8
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:9,the real answer is 9
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:6,the real answer is 6
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:8,the real answer is 8
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:4,the real answer is 4
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:9,the real answer is 9
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:3,the real answer is 3
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:8,the real answer is 8
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:5,the real answer is 5
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:4,the real answer is 4
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:5,the real answer is 5
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:7,the real answer is 7
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:2,the real answer is 2
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:5,the real answer is 5
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:0,the real answer is 0
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:5,the real answer is 5
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:2,the real answer is 2
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:6,the real answer is 6
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:4,the real answer is 4
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:1,the real answer is 1
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:0,the real answer is 0
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:8,the real answer is 8
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:9,the real answer is 9
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:7,the real answer is 7
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:9,the real answer is 9
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:8,the real answer is 8
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:9,the real answer is 9
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:1,the real answer is 1
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:5,the real answer is 5
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:6,the real answer is 6
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:9,the real answer is 9
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:8,the real answer is 8
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:5,the real answer is 5
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:9,the real answer is 9
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:4,the real answer is 4
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:2,the real answer is 2
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:9,the real answer is 9
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:8,the real answer is 8
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:3,the real answer is 3
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:7,the real answer is 7
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:9,the real answer is 9
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:1,the real answer is 1
    the total number of error is 5
    total error rate is: 0.005285 
    the classifier came back with:7,the real answer is 9
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:4,the real answer is 4
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:9,the real answer is 9
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:4,the real answer is 4
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:9,the real answer is 9
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:6,the real answer is 6
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:0,the real answer is 0
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:1,the real answer is 1
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:6,the real answer is 6
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:5,the real answer is 5
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:9,the real answer is 9
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:1,the real answer is 1
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:2,the real answer is 2
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:5,the real answer is 5
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:7,the real answer is 7
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:7,the real answer is 7
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:6,the real answer is 6
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:1,the real answer is 1
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:5,the real answer is 5
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:9,the real answer is 9
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:7,the real answer is 7
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:0,the real answer is 0
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:5,the real answer is 5
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:2,the real answer is 2
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:7,the real answer is 7
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:4,the real answer is 4
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:5,the real answer is 5
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:6,the real answer is 6
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:1,the real answer is 1
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:6,the real answer is 6
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:2,the real answer is 2
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:3,the real answer is 3
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:4,the real answer is 4
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:3,the real answer is 3
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:8,the real answer is 8
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:1,the real answer is 1
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:0,the real answer is 0
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:2,the real answer is 2
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:1,the real answer is 1
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:3,the real answer is 3
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:4,the real answer is 4
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:8,the real answer is 8
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:8,the real answer is 8
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:4,the real answer is 4
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:4,the real answer is 4
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:2,the real answer is 2
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:9,the real answer is 9
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:9,the real answer is 9
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:4,the real answer is 4
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:2,the real answer is 2
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:0,the real answer is 0
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:0,the real answer is 0
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:9,the real answer is 9
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:9,the real answer is 9
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:3,the real answer is 3
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:8,the real answer is 8
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:2,the real answer is 2
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:8,the real answer is 8
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:8,the real answer is 8
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:4,the real answer is 4
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:6,the real answer is 6
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:7,the real answer is 7
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:7,the real answer is 7
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:2,the real answer is 2
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:0,the real answer is 0
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:8,the real answer is 8
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:9,the real answer is 9
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:3,the real answer is 3
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:4,the real answer is 4
    the total number of error is 6
    total error rate is: 0.006342 
    the classifier came back with:1,the real answer is 8
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:1,the real answer is 1
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:2,the real answer is 2
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:5,the real answer is 5
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:1,the real answer is 1
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:2,the real answer is 2
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:7,the real answer is 7
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:2,the real answer is 2
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:2,the real answer is 2
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:2,the real answer is 2
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:3,the real answer is 3
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:5,the real answer is 5
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:2,the real answer is 2
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:5,the real answer is 5
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:9,the real answer is 9
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:5,the real answer is 5
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:6,the real answer is 6
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:7,the real answer is 7
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:8,the real answer is 8
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:8,the real answer is 8
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:8,the real answer is 8
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:8,the real answer is 8
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:0,the real answer is 0
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:4,the real answer is 4
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:7,the real answer is 7
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:4,the real answer is 4
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:1,the real answer is 1
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:5,the real answer is 5
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:4,the real answer is 4
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:3,the real answer is 3
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:3,the real answer is 3
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:9,the real answer is 9
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:7,the real answer is 7
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:4,the real answer is 4
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:5,the real answer is 5
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:9,the real answer is 9
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:7,the real answer is 7
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:2,the real answer is 2
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:3,the real answer is 3
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:1,the real answer is 1
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:9,the real answer is 9
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:2,the real answer is 2
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:4,the real answer is 4
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:8,the real answer is 8
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:3,the real answer is 3
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:9,the real answer is 9
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:7,the real answer is 7
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:2,the real answer is 2
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:5,the real answer is 5
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:8,the real answer is 8
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:6,the real answer is 6
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:1,the real answer is 1
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:7,the real answer is 7
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:4,the real answer is 4
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:9,the real answer is 9
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:6,the real answer is 6
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:9,the real answer is 9
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:8,the real answer is 8
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:5,the real answer is 5
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:4,the real answer is 4
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:5,the real answer is 5
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:5,the real answer is 5
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:4,the real answer is 4
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:6,the real answer is 6
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:1,the real answer is 1
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:5,the real answer is 5
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:0,the real answer is 0
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:0,the real answer is 0
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:0,the real answer is 0
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:4,the real answer is 4
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:7,the real answer is 7
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:0,the real answer is 0
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:4,the real answer is 4
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:4,the real answer is 4
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:8,the real answer is 8
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:4,the real answer is 4
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:1,the real answer is 1
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:7,the real answer is 7
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:5,the real answer is 5
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:3,the real answer is 3
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:1,the real answer is 1
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:1,the real answer is 1
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:7,the real answer is 7
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:4,the real answer is 4
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:5,the real answer is 5
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:2,the real answer is 2
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:4,the real answer is 4
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:0,the real answer is 0
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:8,the real answer is 8
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:6,the real answer is 6
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:5,the real answer is 5
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:1,the real answer is 1
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:3,the real answer is 3
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:4,the real answer is 4
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:6,the real answer is 6
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:7,the real answer is 7
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:8,the real answer is 8
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:6,the real answer is 6
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:8,the real answer is 8
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:6,the real answer is 6
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:4,the real answer is 4
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:7,the real answer is 7
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:4,the real answer is 4
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:1,the real answer is 1
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:2,the real answer is 2
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:1,the real answer is 1
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:7,the real answer is 7
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:2,the real answer is 2
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:2,the real answer is 2
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:8,the real answer is 8
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:4,the real answer is 4
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:4,the real answer is 4
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:5,the real answer is 5
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:3,the real answer is 3
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:9,the real answer is 9
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:3,the real answer is 3
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:1,the real answer is 1
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:0,the real answer is 0
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:9,the real answer is 9
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:9,the real answer is 9
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:5,the real answer is 5
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:0,the real answer is 0
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:1,the real answer is 1
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:3,the real answer is 3
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:1,the real answer is 1
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:9,the real answer is 9
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:3,the real answer is 3
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:9,the real answer is 9
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:9,the real answer is 9
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:7,the real answer is 7
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:9,the real answer is 9
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:3,the real answer is 3
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:8,the real answer is 8
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:7,the real answer is 7
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:4,the real answer is 4
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:4,the real answer is 4
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:1,the real answer is 1
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:1,the real answer is 1
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:2,the real answer is 2
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:5,the real answer is 5
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:2,the real answer is 2
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:0,the real answer is 0
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:5,the real answer is 5
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:3,the real answer is 3
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:8,the real answer is 8
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:9,the real answer is 9
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:1,the real answer is 1
    the total number of error is 7
    total error rate is: 0.007400 
    the classifier came back with:9,the real answer is 3
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:7,the real answer is 7
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:8,the real answer is 8
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:6,the real answer is 6
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:1,the real answer is 1
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:0,the real answer is 0
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:6,the real answer is 6
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:4,the real answer is 4
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:7,the real answer is 7
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:7,the real answer is 7
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:3,the real answer is 3
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:6,the real answer is 6
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:2,the real answer is 2
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:4,the real answer is 4
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:8,the real answer is 8
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:0,the real answer is 0
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:7,the real answer is 7
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:3,the real answer is 3
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:9,the real answer is 9
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:1,the real answer is 1
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:0,the real answer is 0
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:4,the real answer is 4
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:6,the real answer is 6
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:1,the real answer is 1
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:9,the real answer is 9
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:3,the real answer is 3
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:0,the real answer is 0
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:5,the real answer is 5
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:7,the real answer is 7
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:4,the real answer is 4
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:7,the real answer is 7
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:7,the real answer is 7
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:7,the real answer is 7
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:2,the real answer is 2
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:3,the real answer is 3
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:7,the real answer is 7
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:4,the real answer is 4
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:0,the real answer is 0
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:7,the real answer is 7
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:0,the real answer is 0
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:7,the real answer is 7
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:3,the real answer is 3
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:3,the real answer is 3
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:8,the real answer is 8
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:8,the real answer is 8
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:1,the real answer is 1
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:0,the real answer is 0
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:4,the real answer is 4
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:2,the real answer is 2
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:3,the real answer is 3
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:1,the real answer is 1
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:5,the real answer is 5
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:6,the real answer is 6
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:0,the real answer is 0
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:3,the real answer is 3
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:9,the real answer is 9
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:1,the real answer is 1
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:8,the real answer is 8
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:4,the real answer is 4
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:6,the real answer is 6
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:5,the real answer is 5
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:3,the real answer is 3
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:3,the real answer is 3
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:0,the real answer is 0
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:2,the real answer is 2
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:7,the real answer is 7
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:9,the real answer is 9
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:6,the real answer is 6
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:6,the real answer is 6
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:6,the real answer is 6
    the total number of error is 8
    total error rate is: 0.008457 
    the classifier came back with:6,the real answer is 8
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:5,the real answer is 5
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:4,the real answer is 4
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:4,the real answer is 4
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:2,the real answer is 2
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:8,the real answer is 8
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:3,the real answer is 3
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:8,the real answer is 8
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:1,the real answer is 1
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:9,the real answer is 9
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:4,the real answer is 4
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:3,the real answer is 3
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:4,the real answer is 4
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:1,the real answer is 1
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:6,the real answer is 6
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:1,the real answer is 1
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:6,the real answer is 6
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:6,the real answer is 6
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:2,the real answer is 2
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:3,the real answer is 3
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:9,the real answer is 9
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:5,the real answer is 5
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:2,the real answer is 2
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:9,the real answer is 9
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:1,the real answer is 1
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:9,the real answer is 9
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:1,the real answer is 1
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:5,the real answer is 5
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:1,the real answer is 1
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:3,the real answer is 3
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:3,the real answer is 3
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:2,the real answer is 2
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:6,the real answer is 6
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:1,the real answer is 1
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:7,the real answer is 7
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:7,the real answer is 7
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:4,the real answer is 4
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:0,the real answer is 0
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:5,the real answer is 5
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:5,the real answer is 5
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:5,the real answer is 5
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:7,the real answer is 7
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:2,the real answer is 2
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:9,the real answer is 9
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:6,the real answer is 6
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:1,the real answer is 1
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:0,the real answer is 0
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:4,the real answer is 4
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:8,the real answer is 8
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:3,the real answer is 3
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:2,the real answer is 2
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:5,the real answer is 5
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:2,the real answer is 2
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:3,the real answer is 3
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:2,the real answer is 2
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:9,the real answer is 9
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:5,the real answer is 5
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:7,the real answer is 7
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:9,the real answer is 9
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:3,the real answer is 3
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:8,the real answer is 8
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:1,the real answer is 1
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:3,the real answer is 3
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:0,the real answer is 0
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:0,the real answer is 0
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:2,the real answer is 2
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:5,the real answer is 5
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:3,the real answer is 3
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:2,the real answer is 2
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:7,the real answer is 7
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:5,the real answer is 5
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:2,the real answer is 2
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:8,the real answer is 8
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:1,the real answer is 1
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:4,the real answer is 4
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:3,the real answer is 3
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:3,the real answer is 3
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:6,the real answer is 6
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:0,the real answer is 0
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:2,the real answer is 2
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:6,the real answer is 6
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:6,the real answer is 6
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:7,the real answer is 7
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:4,the real answer is 4
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:1,the real answer is 1
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:9,the real answer is 9
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:3,the real answer is 3
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:6,the real answer is 6
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:0,the real answer is 0
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:4,the real answer is 4
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:2,the real answer is 2
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:1,the real answer is 1
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:8,the real answer is 8
    the total number of error is 9
    total error rate is: 0.009514 
    the classifier came back with:7,the real answer is 1
    the total number of error is 10
    total error rate is: 0.010571 
    the classifier came back with:6,the real answer is 6
    the total number of error is 10
    total error rate is: 0.010571 
    the classifier came back with:2,the real answer is 2
    the total number of error is 10
    total error rate is: 0.010571 
    the classifier came back with:0,the real answer is 0
    the total number of error is 10
    total error rate is: 0.010571 
    the classifier came back with:7,the real answer is 7
    the total number of error is 10
    total error rate is: 0.010571 
    the classifier came back with:4,the real answer is 4
    the total number of error is 10
    total error rate is: 0.010571 
    the classifier came back with:7,the real answer is 7
    the total number of error is 10
    total error rate is: 0.010571 
    the classifier came back with:5,the real answer is 5
    the total number of error is 10
    total error rate is: 0.010571 
    the classifier came back with:2,the real answer is 2
    the total number of error is 10
    total error rate is: 0.010571 
    the classifier came back with:9,the real answer is 9
    the total number of error is 10
    total error rate is: 0.010571 
    the classifier came back with:3,the real answer is 3
    the total number of error is 10
    total error rate is: 0.010571 
    the classifier came back with:5,the real answer is 5
    the total number of error is 10
    total error rate is: 0.010571 
    the classifier came back with:0,the real answer is 0
    the total number of error is 10
    total error rate is: 0.010571 
    the classifier came back with:7,the real answer is 7
    the total number of error is 10
    total error rate is: 0.010571 
    the classifier came back with:0,the real answer is 0
    the total number of error is 10
    total error rate is: 0.010571 
    the classifier came back with:3,the real answer is 3
    the total number of error is 10
    total error rate is: 0.010571 
    the classifier came back with:8,the real answer is 8
    the total number of error is 10
    total error rate is: 0.010571 
    the classifier came back with:4,the real answer is 4
    the total number of error is 10
    total error rate is: 0.010571 
    the classifier came back with:4,the real answer is 4
    the total number of error is 10
    total error rate is: 0.010571 
    the classifier came back with:7,the real answer is 7
    the total number of error is 10
    total error rate is: 0.010571 
    the classifier came back with:1,the real answer is 1
    the total number of error is 10
    total error rate is: 0.010571 
    the classifier came back with:4,the real answer is 4
    the total number of error is 10
    total error rate is: 0.010571 
    the classifier came back with:1,the real answer is 8
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:7,the real answer is 7
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:0,the real answer is 0
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:5,the real answer is 5
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:6,the real answer is 6
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:6,the real answer is 6
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:8,the real answer is 8
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:9,the real answer is 9
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:1,the real answer is 1
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:8,the real answer is 8
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:4,the real answer is 4
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:7,the real answer is 7
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:9,the real answer is 9
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:3,the real answer is 3
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:6,the real answer is 6
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:5,the real answer is 5
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:4,the real answer is 4
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:8,the real answer is 8
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:8,the real answer is 8
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:4,the real answer is 4
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:7,the real answer is 7
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:9,the real answer is 9
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:8,the real answer is 8
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:5,the real answer is 5
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:6,the real answer is 6
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:4,the real answer is 4
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:3,the real answer is 3
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:7,the real answer is 7
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:8,the real answer is 8
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:9,the real answer is 9
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:0,the real answer is 0
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:5,the real answer is 5
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:3,the real answer is 3
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:3,the real answer is 3
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:5,the real answer is 5
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:4,the real answer is 4
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:6,the real answer is 6
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:4,the real answer is 4
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:4,the real answer is 4
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:2,the real answer is 2
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:5,the real answer is 5
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:6,the real answer is 6
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:3,the real answer is 3
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:4,the real answer is 4
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:2,the real answer is 2
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:3,the real answer is 3
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:9,the real answer is 9
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:5,the real answer is 5
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:4,the real answer is 4
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:2,the real answer is 2
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:3,the real answer is 3
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:9,the real answer is 9
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:4,the real answer is 4
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:1,the real answer is 1
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:1,the real answer is 1
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:1,the real answer is 1
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:0,the real answer is 0
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:1,the real answer is 1
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:8,the real answer is 8
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:1,the real answer is 1
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:0,the real answer is 0
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:0,the real answer is 0
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:0,the real answer is 0
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:6,the real answer is 6
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:7,the real answer is 7
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:8,the real answer is 8
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:5,the real answer is 5
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:4,the real answer is 4
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:3,the real answer is 3
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:2,the real answer is 2
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:2,the real answer is 2
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:7,the real answer is 7
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:3,the real answer is 3
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:4,the real answer is 4
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:0,the real answer is 0
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:2,the real answer is 2
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:8,the real answer is 8
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:6,the real answer is 6
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:5,the real answer is 5
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:8,the real answer is 8
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:0,the real answer is 0
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:2,the real answer is 2
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:3,the real answer is 3
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:7,the real answer is 7
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:7,the real answer is 7
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:7,the real answer is 7
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:5,the real answer is 5
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:8,the real answer is 8
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:7,the real answer is 7
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:6,the real answer is 6
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:4,the real answer is 4
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:4,the real answer is 4
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:1,the real answer is 1
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:4,the real answer is 4
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:9,the real answer is 9
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:3,the real answer is 3
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:0,the real answer is 0
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:0,the real answer is 0
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:8,the real answer is 8
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:8,the real answer is 8
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:6,the real answer is 6
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:3,the real answer is 3
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:2,the real answer is 2
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:9,the real answer is 9
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:3,the real answer is 3
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:2,the real answer is 2
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:9,the real answer is 9
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:8,the real answer is 8
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:2,the real answer is 2
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:1,the real answer is 1
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:8,the real answer is 8
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:8,the real answer is 8
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:0,the real answer is 0
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:7,the real answer is 7
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:8,the real answer is 8
    the total number of error is 11
    total error rate is: 0.011628 
    the classifier came back with:0,the real answer is 0
    the total number of error is 11
    total error rate is: 0.011628 

