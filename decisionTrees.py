from math import log

#calculate entropy
#entropy 的计算只与输入矩阵的label有关
def clacshaannonEnt(dataSet):
	#这里dataset 是二维数组
	numEntryes = len(dataSet)
	labelCounts={}
	for featVec in dataSet:
		currentLable = featVec[-1]
		if currentLable not in labelCounts.keys():
			labelCounts[currentLable] = 0
		labelCounts[currentLable] +=1
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntryes
		shannonEnt-=prob*log(prob,2)
	return shannonEnt


def createDataSet():
       dataSet = [[1, 1, 'yes'],[1, 1, 'yes'],[1, 0, 'no'],[0, 1, 'no'],[0, 1, 'no']]
       labels = ['no surfacing','flippers']
       return dataSet, labels
# you need to measure the entropy, split the dataset, measure the entropy
# used to find the sub-dataset according to the axis and value. for k in index, if k's value equals value(inputted),put this line into new-set
def splitDataset(dataSet,axis,value):
	retDataSet=[]
	for featVec in dataSet:
		if featVec[axis]==value:
			reducedFeatVec=featVec[:axis]
			#extend 和 append是不一样的
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet

# find the best frature 
#对于输入的dataset，对于每一个feature计算entropy，和 informationGain，找到最大的，返回feature
#informationGain的数据与概率有关
def chooseBeseFeatureToSplit(dataset):
	numFeatures = len(dataset[0])-1
	baseEntropy = clacshaannonEnt(dataset)
	bestInfoGain=0
	bestFeature = -1
	for i in range(numFeatures):
		featList = [exampel[i] for exampel in dataset]
		uniqueVals = set(featList)
		newEntry = 0.0
		for value in uniqueVals:
			subDataset =  splitDataset(dataset,i,value)
			prob = len(subDataset)/len(dataset)
			newEntry = prob*clacshaannonEnt(subDataset)
		infoGain = baseEntropy-newEntry
		if(infoGain > bestInfoGain):
			bestInfoGain =infoGain
			bestFeature=i
	return bestFeature
#当所有features都分类完后，如何还有叶子结点中不纯，用多数表决法
def majorutyCnt(classList):
	classCount={}
	for vote in classList:
		if vote not in classCount.keys(): 
			classCount[vote]=0
		classCount[vote]+=1
	sortedClassCount = sorted(classCount.items(),keys = classCount.itemgetter(1),reverse = True)
	return sortedClassCount[0][0]
# 本身创建树的过程就是一个循环过程。
#如果已经处理完了所有feature但不纯则多数表决，或者当前已经是纯的则停止
#如果都没有，则找到最适合的feature，然后分类后再分类
def createTree(dataset,labels):
	classList=[example[-1] for example in dataset]
	if classList.count(classList[0])==len(classList):
		return classList[0]
	if len(dataset)==1:
		return majorutyCnt(dataset)
	bestFeat = chooseBeseFeatureToSplit(dataset)
	bestFeatlabel = labels[bestFeat]
	myTree = {bestFeatlabel:{}}
	del(labels[bestFeat])
	featValues = [example[bestFeat] for example in dataset]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		sublabel = labels[:]
		myTree[bestFeatlabel][value] = createTree(splitDataset(dataset,bestFeat,value),sublabel)
	return myTree





if __name__=='__main__':
	myDat,label = createDataSet()
	# print(chooseBeseFeatureToSplit(myDat))
	# print(clacshaannonEnt(myDat))
	myTree = createTree(myDat,label)
	print(myTree)


