import numpy as np
import utils
import math
from numpy.random import seed as seed
from numpy.random import rand as rand
from utils import MyUtils
import pla


def main():
    print("Testing Z-Transforms")
    passed = test_transforms()
    if passed:
        print("Testing PLA")
        test_pla()
    else:
        print("Skipping PLA Test")

####################################################################################
# PLA TESTING
####################################################################################

def test_pla():
    passed = np.full((2,2), False)
    expected_errors = np.array([0,0,9,12]).reshape(2,2)

    for i, pocket in enumerate([False, True]):
        for j, degree in enumerate([1,2]):
            errors = _test_pla(degree, pocket, seed_num=11)
            passed[i,j] = expected_errors[i,j] == errors

    if passed.all(): print("PLA: SUCCESS!!!")
    else:
        print("PLA: Please check failed cases.")
        degrees = np.array([1,2])
        for idx in range(2):
            if not passed[idx].all():
                #Output cases failed for each method
                failed = [not elem for elem in passed[idx]]
                print(f"pocket method: {idx == 1}, failed for degree = {degrees[failed]}")
        return False
    
# a function that  generates a w_star, the classifer that the PLA will try to find
def w_star_gen_2d(degree = 1, seed_num = None):
    ''' only suppot degree <= 4 for demo purpose. 
        Don't have time to code out the generic form for any degree.
        A seed, if provided, will set a random seed for numpy to use in generating weights.
    '''
    if seed_num: seed(seed_num)

    if degree == 1:
        w_star = rand(3,1) * np.sign(rand(3,1)-0.5)
    elif degree == 2:
        w_star = rand(6,1) * np.sign(rand(6,1)-0.5)
    elif degree == 3:
        w_star = rand(10,1) * np.sign(rand(10,1)-0.5)
    elif degree == 4:
        w_star = rand(15,1) * np.sign(rand(15,1)-0.5)
    
    else:
        print('Z space degree is too high. Not supported now\n')
        quit()

    return w_star

def _test_pla(degree = 1, pocket = False, seed_num = None, n = 400):
    w_star = w_star_gen_2d(degree = degree, seed_num = seed_num)

    seed(seed_num)
    # randomly generate the features of n samples. 
    X = rand(n,2) 
    Z = MyUtils.z_transform(X, degree = degree)
    Z = np.insert(Z, 0, 1, axis = 1)  # add the bias column

    #create the label array
    y = np.sign(Z @ w_star)
    to_delete = [] #index of rows to be deleted because those points sit on w_star
    for i in range(n):
        if y[i] == 0:
            to_delete.append(i)

    # delete those samples sitting right on w_star        
    X = np.delete(X, to_delete, axis = 0)
    Z = np.delete(Z, to_delete, axis = 0)
    y = np.delete(y, to_delete, axis = 0)
    n = n - len(to_delete)

    # mess up mess_percent of the samples to test pocket pla   
    mess_percent = 0.02
    if pocket: 
        for i in range(math.floor(n * mess_percent)):
            pos = math.floor(rand()*n)
            y[pos] = y[pos] * (-1)

    perceptron = pla.PLA(degree=degree)

    if pocket: 
        perceptron.fit(X, y, pocket = True, epochs = 500)
    else: 
        perceptron.fit(X, y, pocket = False)
    return perceptron.error(X,y)

####################################################################################
# Z-TRANSFORMATION TESTING
####################################################################################

def test_transforms():
    n = 3
    max_d = 3
    max_r = 5
    count = 0
    data = getExpectedResults()
    passed = np.full((max_d, max_r), False)
    for d in range(1,max_d+1):
        for r in range(1,max_r+1):
            passed[d-1,r-1] = test_single_transform(n,d,r,data,count)
            count += 1
    if passed.all():
        print("Z-Transformation: SUCCESS!!!")
        return True
    else:
        print("Z-Transformation: Please check failed cases.")
        all_r = np.array([1,2,3,4,5])
        for d in range(max_d):
            if not passed[d].all():
                #Output cases failed for each column
                failed = [not elem for elem in passed[d]]
                print(f"d = {d}, failed for r = {all_r[failed]}")
        return False

def test_single_transform(n,d,r,data, data_idx):
    X = generateX(n,d)
    Z = utils.MyUtils.z_transform(X,degree=r)
    num_columns = combinations(r,d)
    if num_columns != Z.shape[1]:
        print(f"An incorrect number of columns were generated for degree {r} and {d} feature(s).\nExpected: {num_columns}\nFound: {Z.shape[1]}")
        return False
    if not np.all(data[data_idx] == Z):
        print(f"Mismatched Z matrices for degree {r} and {d} feature(s).\nExpected:\n{data[data_idx]}\nFound:\n{Z}")
        return False
    return True

def combinations(r,d):
    sum = 0
    for i in range(r):
        sum += math.comb(i+d,d-1)
    return sum

def generateX(n,d):
    row = np.arange(1,d + 1).reshape(1,d) #create the first row
    X = row.copy()
    for i in range(1,n):
        X = np.append(X, row + i, axis=0)
    return X

def getExpectedResults(file="output.npz"):
    container = np.load(file)
    data = [container[key] for key in container]
    return data

if __name__ == '__main__':
    main()