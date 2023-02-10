import numpy as np
import math
import linear_regression as LR

def main():
    lr = LR.LinearRegression()
    test_X1 = np.array([[1,2],[2,3],[3,4]])
    test_y1 = np.array([[1],[0],[-1]])
    # print(test_X1,"\n\n",test_y1)
    # print(np.shape(test_X1))
    # lr._fit_cf(test_X1, test_y1)
    lr._fit_gd(test_X1,test_y1)

if __name__ == '__main__':
    main()