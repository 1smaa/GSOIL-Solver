import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt

PATH="parameters.json"
CWD=os.getcwd()
with open(PATH,mode="r",encoding="utf-8") as f:
    PARAM=json.load(f)
SIM=int(1e5)

class StandardPropagation(object):
    def __init__(self,init: tuple[float,float,float],dt: float,params: dict,i: np.ndarray) -> None:
        self._i=0
        self._n,self._s,self._phi=init
        self._dt=dt
        self._params=params
        self._curr=i
        
    def step(self) -> None:
        if(self._i==len(self._curr)):
            print(self._i)
            raise ValueError("Stepped too far.")
        n_new=self._n+self._dt*(self._curr[self._i]/(self._params["q"])-(self._n/self._params["tau_n"])-((self._params["g"]*(self._n-self._params["N_TR"])*self._s)/(1+(self._params["e"]*self._s))))
        s_new=self._s+self._dt*((self._params["g"]*(self._n-self._params["N_TR"])*self._s/(1+self._params["e"]*self._s))-(self._s/self._params["tau_p"])+(self._params["beta"]*self._n/self._params["tau_n"]))
        phi_new=self._phi+self._dt*self._params["alpha"]*0.5*(self._params["g"]*(self._n-self._params["N_TR"]))
        self._n=n_new
        self._s=s_new
        self._phi=phi_new
        self._i+=1
        
    def get(self) -> tuple[float,float,float]:
        return self._n,self._s,self._phi
    
    def get_index(self) -> int:
        return self._i

class SecondaryPropagation(object):
    def __init__(self,prop: StandardPropagation,init: tuple[float,float,float],dt: float,params: dict,i: np.ndarray) -> None:
        self._primary=prop
        self._i=0
        self._n,self._s,self._phi=init
        self._dt=dt
        self._params=params
        self._curr=i
        
    def step(self) -> None:
        if(self._i==len(self._curr)):
            print(self._i)
            raise ValueError("Stepped too far.")
        master_n,master_s,master_phi=self._primary.get()
        dn_dt=self._curr[self._i]/(self._params["q"])-(self._n/self._params["tau_n"])-((self._params["g"]*(self._n-self._params["N_TR"])*self._s)/(1+(self._params["e"]*self._s)))
        n_new=self._n+self._dt*dn_dt
        ds_dt_free=(self._params["g"]*(self._n-self._params["N_TR"])*self._s/(1+self._params["e"]*self._s))-(self._s/self._params["tau_p"])+(self._params["beta"]*self._n/self._params["tau_n"])
        ds_dt_oil=2*self._params["kappa"]*np.sqrt(master_s*self._s)*np.cos(self._phi-master_phi-(self._params["detuning"]*self._i*self._dt))
        ds_dt=ds_dt_free+ds_dt_oil
        s_new=self._s+self._dt*ds_dt
        dphi_dt_free=self._params["alpha"]*0.5*((self._params["GAMMA"]*self._params["g"]*(self._n-self._params["N_TR"])))
        dphi_dt_oil=-self._params["kappa"]*np.sqrt(master_s/self._s)*np.sin(self._phi-master_phi-(self._params["detuning"]*self._i*self._dt))
        dphi_dt=dphi_dt_free+dphi_dt_oil
        phi_new=self._phi+self._dt*dphi_dt
        self._n=n_new
        self._s=s_new
        self._phi=phi_new
        self._primary.step()
        self._i+=1
        
    def get(self) -> tuple[float,float,float]:
        return self._n,self._s,self._phi
    
    def get_index(self) -> int:
        return self._i
        

def make_pulse_train(period: float, duration: float, peak: float, bias: float, delay: float = 0.0) -> np.ndarray:
    """
    Generate a rectangular pulse train with period, duration, peak, bias, and initial delay.
    """
    dt = PARAM["DT"]
    t = np.arange(SIM) * dt
    shifted = t - delay
    phase = np.mod(shifted, period)
    return np.where((shifted >= 0) & (phase < duration), bias + peak, bias)
    

def main() -> None:
    I=make_pulse_train(44000*PARAM["DT"],8000*PARAM["DT"],4e-3,5e-4,1200*PARAM["DT"])
    prop=StandardPropagation((PARAM["N0"],PARAM["S0"],0),PARAM["DT"],PARAM,I)
    s_hist=[]
    n_hist=[]
    I_hist=[]
    for j in range(SIM):
        prop.step()
        n,s,p=prop.get()
        n_hist.append(n)
        s_hist.append(s)
        I_hist.append(I[j])
    # assume prop.history exists (or build arrays as you run)
    t = np.linspace(0,PARAM["DT"]*SIM,SIM)

    # pick region right after pulse (adjust indices to your pulse location)
    idx0 = 200   # start index for inspection (ps scale depends on DT)
    idx1 = 260

    print("N around pulse:", max(n_hist[idx0:idx1]), min(n_hist[idx0:idx1]))
    print("S around pulse:", max(s_hist[idx0:idx1]), min(s_hist[idx0:idx1]))
    print("I around pulse:", max(I_hist[idx0:idx1]))

    # compute instantaneous derivatives (finite differences)
    dn_dt = np.diff(n_hist) / PARAM["DT"]
    ds_dt = np.diff(s_hist) / PARAM["DT"]

    print("ds_dt just after pulse (first 10):", ds_dt[idx0:idx0+10])
    print("dn_dt just after pulse (first 10):", dn_dt[idx0:idx0+10])
    j = np.argmax(s_hist) + 1
    Gamma = PARAM["GAMMA"]
    g = PARAM["g"]
    N_TR = PARAM["N_TR"]
    eps = PARAM["e"]
    tau_p = PARAM["tau_p"]

    net_gain = Gamma * g * (n_hist[j] - N_TR) / (1.0 + eps * s_hist[j]) - 1.0 / tau_p
    print("net_gain at t+1:", net_gain)
    # plot a zoom to inspect
    plt.figure()
    plt.subplot(3,1,1); plt.plot(t,n_hist); plt.title('N')
    plt.subplot(3,1,2); plt.plot(t,s_hist); plt.title('S')
    plt.subplot(3,1,3); plt.plot(t,I_hist); plt.title('I')
    plt.show()

    plt.show()

if __name__=="__main__":
    main()