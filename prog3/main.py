from pla import *
from utils import MyUtils

def plaMain():
    ex1 = [[1],[2],[3]]
    ex2 = [[1, 2], [2, 3], [3, 4]]
    ex3 = [[1,2,3],[2,3,4],[3,4,5]]

    y = [1,1,1]

    X = ex3
    myPLA = PLA(5)
    myPLA.fit(X,y,True,100)

if __name__ == "__main__":
    plaMain()