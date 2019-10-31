import matplotlib.pyplot as plt
import numpy as np 
import operator

# No      featueA          featueB         what(A/B)
#  1        3               104               a
#  2        2               100               a
#  3        1                81               a
#  4       100               10               b
#  5       101                4               b
#  6        81                2               b
#  7        18                90              ?

# wirte some compare data 
x1 = np.array([3,2,1])
x2 = np.array([100,101,120])
y1 = np.array([104,100,81])
y2 = np.array([10,4,2])

# unknown data
x= np.array([18])
y= np.array([90])

# draw graph
pointg1 = plt.scatter(x1,y1,c='red')
pointg2 = plt.scatter(x2,y2,c='blue')
pointg3 = plt.scatter(x,y,c='k')
plt.legend(handles=[pointg1,pointg2,pointg3],labels=['lablelA','lablelB','unknown'],loc='best')
# plt.show()

# start Knn
# classification know data
dataset =np.array([[3,104],
                 [2,100],
                 [1,81],
                 [100,10],
                 [101,4],
                 [120,2]])
labels =np.array(['a','a','a','b','b','b'])
unknown_test = np.array([18,90])

def knnfunction(dataset,labels,unknown_test):
    # Euclidean Distance
    # caculation the number of the example 
    x_size = dataset.shape[0]
    # print(x_size)
    # copy unknown data and set the compare time 
    copy=np.tile(unknown_test,(x_size,1)) 
    # caculation different value
    diffMat =copy- dataset
    # caculation standard deviation
    spDiffMat =diffMat**2
    # sum up
    sqDistance =spDiffMat.sum(axis=1)
    # extraction of a root
    distances =sqDistance**0.5
    # sort of the number 
    sortedDistances = distances.argsort()
    classCount ={}
    # set the K value
    k=5
    for i in range(k):
        # get the labels
        votelabel = labels[sortedDistances[i]]
        # sum up the labels number
        classCount[votelabel]= classCount.get(votelabel,0)+1
    sortedClassCount =sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


print(knnfunction(dataset,labels,unknown_test))
