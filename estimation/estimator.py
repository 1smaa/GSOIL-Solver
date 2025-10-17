import numpy as np
from sklearn.linear_model import LinearRegression

class Estimator(object):
    def __init__(self,N0: float,s: np.ndarray,i: np.ndarray,dt: float) -> None:
        self._X=np.column_stack([N0,s[0]])
        self._s=s
        self._dt=dt
        self._i=i
        self._index=0
        if self._s.shape!=self._i.shape:
            raise ValueError("S and I arrays have different shapes")
        
    def step_regression(self) -> np.ndarray:
        regS=LinearRegression().fit(self._X,self._s)
    