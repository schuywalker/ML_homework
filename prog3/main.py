from pla import *
from utils import MyUtils

def plaMain():
    ex1 = [[1],[2],[3]]
    ex2 = [[1, 2], [2, 3], [3, 4]]
    ex3 = [[1,2,3],[2,3,4],[3,4,5]]
    data_idx1 = [[1, 1], [2, 4], [3, 9]]

    y = np.array([1,1,1])
    y = np.reshape(y,(len(y), 1))
    # print("y: ",y)

    X = ex2
    myPLA = PLA(3)
    myPLA.fit(X,y,True,100)

if __name__ == "__main__":
    plaMain()