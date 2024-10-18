import math
import unittest

def get_threshold_cos(u, ratio, min_cos):
    '''
    u: float, control the threshold
    p: float, the ratio of the gradient, last gradient divide current gradient
    min_cos: float, if the cos value is less than min_cos, retu   
    '''  
    sina2 = u**2 * (1 + ratio**2)
    if sina2 > 1:
        return min_cos
    
    cos = math.sqrt(1 - sina2)
    return cos if cos > min_cos else min_cos



    

if __name__ == '__main__':
    a = get_threshold_cos(0.80, 0.85, 0.2)
    print(a)