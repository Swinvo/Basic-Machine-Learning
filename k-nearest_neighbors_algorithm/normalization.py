import numpy as np
# No   featueA          featueB       featureC
# 1       40920           8.3             0.9
# 2       14488           7.1             1.6
# 3       26052           1.4             0.8
# 4       75136           13.1            0.42


dataSet =np.array([[40920,8.3,0.9],
                 [14488,7.1,1.6],
                 [26052,1.4,0.8],
                 [75136,13.1,0.42]])

def normalization (dataSet):
    # min number in dataSet
    min_Val = dataSet.min(0)
     # max number in dataSet
    max_Val = dataSet.max(0)
    ranges = max_Val -min_Val
    # Return a new array of given shape and type, filled with zeros.
    norDataSet = np.zeros(np.shape(dataSet))
    #  how many data 
    m = dataSet.shape[0]
    # caculation the oldValue-min
    norDataSet = dataSet- np.tile(min_Val,(m,1))
    # caculation newValue
    norDataSet = norDataSet/np.tile(ranges,(m,1))
    return norDataSet,ranges,min_Val

print(normalization(dataSet))

