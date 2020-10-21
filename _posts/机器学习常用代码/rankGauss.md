- https://zhuanlan.zhihu.com/p/184630770
- https://github.com/RogerZhangsc/Porto-Seguros-Safe-Driver-Prediction/blob/master/rgn/rgn.py
- https://github.com/blueloveTH/rankGaussian_python/blob/master/rankGaussian.ipynb
- https://mp.weixin.qq.com/s?__biz=MzU1Nzc1NjI0Nw==&mid=2247485922&idx=1&sn=38709e304f33fb54455a5c844ee4a9e1&chksm=fc31b3dacb463acce9e753da7e261ea5798fc84081c6f1fa7cdf411219024b490a8b69886829&mpshare=1&scene=1&srcid=1009bQArjmN6jg3IlBr12E01&sharer_sharetime=1603179802870&sharer_shareid=9a422852ec6315623c1df8d624504d2e&key=982bd9ff62fabdf7602cb147c54bde5f69be3a1a49b2ea792d73219bc8e68e39a30d40df648c597c27902d48ea79833d07622c49b5b7cb219bef29e5bd817baf7c99c5761c90572f6b508ed88bba467a3e69ceecc410860bdfb7999c843b47f15686256dbcbd90841d55eb7d60cadcde6047eeab9ba616e65d530c97e110cb5c&ascene=1&uin=MjE5NjM3MzgwMQ%3D%3D&devicetype=Windows+10+x64&version=6300002f&lang=zh_CN&exportkey=AY2cxc%2FGX%2BdaqoVHKHsH138%3D&pass_ticket=o5wDY7K9quXf9vH%2BQxRqR3mhtytaGPDGrJT57S8BQfstEj7SmBT9ENiqA9UDA%2FFu&wx_header=0


```python
import numpy as np
from collections import Counter, OrderedDict

class RGN:
    '''Rank Gaussian Normalization'''
    def __init__(self, data=None, precision=np.float32):
        #data: 1D array or list
        self._data = data
        self.precision = precision        
        self._output = None
        if self._data is None:
            self._trafo_map = None
        else:
            self.fit_transform(self._data)
    
    @property
    def data(self):
        return self._data
    
    @property
    def output(self):
        return self._output
    
    @property
    def precision(self):
        return self._precision
    
    @precision.setter
    def precision(self, p):
        if not isinstance(p, type):
            raise ValueError('precision must be a data type, e.g.: np.float64')
        self._precision = p
    
    def _RationalApproximation(self, t:float)->float: 
        c = [2.515517, 0.802853, 0.010328]
        d = [1.432788, 0.189269, 0.001308]
        return t - ((c[2]*t + c[1])*t + c[0]) / (((d[2]*t + d[1])*t + d[0])*t + 1.0)

    def _NormalCDFInverse(self, p:float) -> float:

        if (p <= 0.0 or p >= 1.0):
            raise Exception('0<p<1. The value of p was: {}'.format(p))
        if (p < 0.5):
            return -self._RationalApproximation(np.sqrt(-2.0*np.log(p)) )
        return self._RationalApproximation( np.sqrt(-2.0*np.log(1-p)) )

    def _vdErfInvSingle01(self, x:float) -> float:
        if x == 0:
            return 0
        elif x < 0:
            return -self._NormalCDFInverse(-x)*0.7
        else:
            return self._NormalCDFInverse(x)*0.7
    
    def fit_transform(self, dataIn:list) -> dict:
        self.fit(dataIn)
        return self.transform(dataIn)

    def fit(self, dataIn:list):
        self._data = dataIn
        trafoMap = OrderedDict()
        hist = Counter(dataIn)
        if len(hist) == 0:
            pass
        elif len(hist) == 1:
            key = list(hist.keys())[0]
            trafoMap[key] = 0.0
        elif len(hist) == 2:
            keys = sorted(list(hist.keys()))
            trafoMap[keys[0]] = 0.0
            trafoMap[keys[1]] = 1.0
        else:
            N = cnt = 0
            for it in hist:
                N += hist[it]
            assert (N == len(dataIn))
            mean = 0.0
            for it in sorted(list(hist.keys())):
                rankV = cnt / N
                rankV = rankV * 0.998 + 1e-3
                rankV = self._vdErfInvSingle01(rankV)
                assert(rankV >= -3.0 and rankV <= 3.0)
                mean += hist[it] * rankV
                trafoMap[it] = rankV
                cnt += hist[it]
            mean /= N
            for it in trafoMap:
                trafoMap[it] -= mean
        self._trafo_map = trafoMap
        return 

    def _binary_search(self, keys, val):
        start, end = 0, len(keys)-1
        while start+1 < end:
            mid = (start + end) // 2
            if val < keys[mid]:
                end = mid
            else:
                start = mid
        return keys[start], keys[end]

    def transform(self, dataIn:list) -> dict:
        dataOut = []
        trafoMap = self._trafo_map
        keys = list(trafoMap.keys())
        if len(keys) == 0:
            raise Exception('No transfermation map')
        for i in range(len(dataIn)):
            val = dataIn[i]
            trafoVal = 0.0
            if val <= keys[0]:
                trafoVal = trafoMap[keys[0]]
            elif val >= keys[-1]:
                trafoVal = trafoMap[keys[-1]]
            elif val in trafoMap:
                trafoVal = trafoMap[val]
            else:
                lower_key, upper_key = self._binary_search(keys, val)
                x1, y1 = lower_key, trafoMap[lower_key]
                x2, y2 = upper_key, trafoMap[upper_key]

                trafoVal = y1 + (val - x1) * (y2 - y1) / (x2 - x1)
            dataOut.append(trafoVal)
        dataOut = np.asarray(dataOut, dtype=self.precision)
        self._output = dataOut
        return self._output  

if __name__ == '__main__':
    data = [-19.9378,10.5341,-32.4515,33.0969,24.3530,-1.1830,-1.4106,-4.9431,
        14.2153,26.3700,-7.6760,60.3346,36.2992,-126.8806,14.2488,-5.0821,
        1.6958,-21.2168,-49.1075,-8.3084,-1.5748,3.7900,-2.1561,4.0756,
        -9.0289,-13.9533,-9.8466,79.5876,-13.3332,-111.9568,-24.2531,120.1174]
    rgn = RGN(data)
    print(rgn.output)
```