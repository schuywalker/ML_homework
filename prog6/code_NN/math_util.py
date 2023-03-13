# Place your EWU ID and name here

## delete the `pass` statement in every function below and add in your own code. 


import numpy as np



# Various math functions, including a collection of activation functions used in NN.




class MyMath:

    def tanh(x):
        ''' tanh function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is tanh of the corresponding element in array x
        '''
        # input array. lazy approach is tanh function from tanh. x = np.array(x). then tanh.
        # can code out later - he had on two separate lines
        return np.tanh(np.array(x))

    
    def tanh_de(x):
        ''' Derivative of the tanh function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is tanh derivative of the corresponding element in array x
        '''
        # find partial deriv. dy/dx for y = tanh(x).  # google derivative of tanh.
        # 1 - [tanh(x)] ^ 2
        # make x and array, get tanh of x, and then 1.0 - temp * temp
        x = np.array(x)
        tx = np.tanh(x)
        return (1 - (tx * tx))

    
    def logis(x):
        ''' Logistic function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is logistic of 
                    the corresponding element in array x
        '''
        x = np.array(x)
        xx = np.divide(x, 2.0)
        xxx = np.tanh(xx) + 1.0
        ret = np.divide(xxx,2.0)
        return ret
    # lazy way: rewrite tanh(x) as a function of x. google sigmoid function as a function of tanh
    # algebra out the tanh(x) = on slide 17. 1/2(tanh(x/2)+1)

    
    def logis_de(x):
        ''' Derivative of the logistic function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is logistic derivative of 
                    the corresponding element in array x
        '''
        return np.divide((MyMath.tanh_de(np.divide(x,2.0))), 4.0)
# deriv of logis.1/4(tanh'(x/2)) = sig'(x)
# just called func you already wrote
# turn to array
# MyMath.tanh_de(x/2) / 4


    
    def iden(x):
        ''' Identity function
            Support vectorized operation
            
            x: an array type of real numbers
            return: the numpy array where every element is the same as
                    the corresponding element in array x
        '''
        x = np.array(x)
        return x

    
    def iden_de(x):
        ''' The derivative of the identity function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array of all ones of the same shape of x.
        '''
        # derivative of iden(x) = x is always 1 (straight diagonal line). but it can be a vector.
        x = np.array(x)
        return np.ones(x.shape)
        

    def relu(x):
        ''' The ReLU function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is the max of: zero vs. the corresponding element in x.
        '''
        # returns x if positive ( > 0), else 0.
        # not differentiable. people choose 1 for deriv at 0.
        x = np.array(x)
        v_max = np.vectorize(max)
        return v_max(x,0)



    
    def _relu_de_scaler(x):
        ''' The derivative of the ReLU function. Scaler version.
        
            x: a real number
            return: 1, if x > 0; 0, otherwise.
        '''
        return (1 if (x > 0) else 0)
    # private function. can vectorize for relu_de's use if you want.

    
    def relu_de(x):
        ''' The derivative of the ReLU function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is the _relu_de_scaler of the corresponding element in x.   
        '''
        x = np.array(x)
        return np.where(x > 0, 1, 0)

    