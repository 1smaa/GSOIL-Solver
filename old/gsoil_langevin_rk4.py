import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt
from scipy.constants import h,c

PATH="parameters.json"
CWD=os.getcwd()
with open(PATH,mode="r",encoding="utf-8") as f:
    PARAM=json.load(f)
SIM=int(1e4)

BIAS=4e-3
PEAK=25e-3

SANITY_N=1e40
SANITY_S=1e40
#Threshold current 11.5mA

class StandardPropagation(object):
    def __init__(self,init: tuple[float,float,float],dt: float,params: dict,i: np.ndarray) -> None:
        self._i=0
        self._n,self._s,self._phi=init
        self._dt=dt
        self._params=params
        self._curr=i
        self._rng=np.random.default_rng()
        
    def _derivatives(self, n, s, phi, i_val):
        sat = (self._params["e"] * s) / (1 + self._params["e"] * s)

        dn_dt = i_val / self._params["q"] - (n / self._params["tau_n"]) - (
           (self._params["g"] / self._params["e"]) * (n - self._params["N_TR"]) * sat
        )
        ds_dt = (
            ((self._params["g"] / self._params["e"]) * (n - self._params["N_TR"]) * sat)
            - (s / self._params["tau_p"])
            + (self._params["beta"] * n / self._params["tau_n"])
        )
        dphi_dt = self._params["alpha"] * 0.5 * (self._params["g"] * (n - self._params["N_TR"]))
        return dn_dt, ds_dt, dphi_dt

    def step(self) -> None:
        if self._i == len(self._curr):
            raise ValueError("Stepped too far.")

        i_val = self._curr[self._i]
        n, s, phi = self._n, self._s, self._phi
        dt = self._dt

        # --- RK4 integration ---
        k1 = self._derivatives(n, s, phi, i_val)
        k2 = self._derivatives(n + 0.5 * dt * k1[0], s + 0.5 * dt * k1[1], phi + 0.5 * dt * k1[2], i_val)
        k3 = self._derivatives(n + 0.5 * dt * k2[0], s + 0.5 * dt * k2[1], phi + 0.5 * dt * k2[2], i_val)
        k4 = self._derivatives(n + dt * k3[0], s + dt * k3[1], phi + dt * k3[2], i_val)

        self._n = max(n + (dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), 0)
        self._s = max(s + (dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]), 1)
        self._phi = phi + (dt / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])

        # add Langevin noise
        self._langevin_noise()
        self._i += 1
        
    def _langevin_noise(self) -> None:
        def F_S() -> float:
            coeff=(2*self._params["beta"]*self._n*self._s)/(self._params["tau_p"])
            if coeff<0: return 0
            return np.sqrt(coeff)*self._rng.standard_normal(1)[0]*np.sqrt(1/self._dt)
        def F_PHI() -> float:
            coeff=(self._params["beta"]*self._n)/(2*self._params["tau_p"]*self._s)
            if coeff<0: return 0
            return np.sqrt(coeff)*self._rng.standard_normal(1)[0]*np.sqrt(1/self._dt)
        def F_Z() -> float:
            coeff=(2*self._n)/(self._params["V"]*self._params["tau_n"])
            if coeff<0: return 0
            return np.sqrt(coeff)*self._rng.standard_normal(1)[0]*np.sqrt(1/self._dt)
        def F_N(F_Z: float,F_S: float) -> float:
            return F_Z-(F_S/self._params["GAMMA"])
        fs,fphi,fz=F_S(),F_PHI(),F_Z()
        fn=F_N(fz,fs)
        self._n=min(SANITY_N,self._n+fn)
        self._s=min(SANITY_S,self._s+fs)
        self._phi+=fphi
        
    def get(self) -> tuple[float,float,float]:
        return self._n,self._s,self._phi
    
    def get_index(self) -> int:
        return self._i

class SecondaryPropagation(StandardPropagation):
    def __init__(self,prop: StandardPropagation,init: tuple[float,float,float],dt: float,params: dict,i: np.ndarray) -> None:
        self._primary=prop
        self._i=0
        self._n,self._s,self._phi=init
        self._dt=dt
        self._params=params
        self._curr=i
        self._rng=np.random.default_rng()
        
    def _derivatives(self, n, s, phi, i_val, master_n, master_s, master_phi, idx):
        dn_dt = i_val / self._params["q"] - (n / self._params["tau_n"]) - (
            (self._params["g"] * (n - self._params["N_TR"]) * s) / (1 + (self._params["e"] * s))
        )
        ds_dt_free = (
            (self._params["g"] * (n - self._params["N_TR"]) * s / (1 + self._params["e"] * s))
            - (s / self._params["tau_p"])
            + (self._params["beta"] * n / self._params["tau_n"])
        )
        ds_dt_oil = 2 * self._params["kappa"] * np.sqrt(max(master_s * s, 0)) * np.cos(
            phi - master_phi - (self._params["detuning"] * idx * self._dt)
        )
        ds_dt = ds_dt_free + ds_dt_oil

        dphi_dt_free = self._params["alpha"] * 0.5 * (self._params["g"] * (n - self._params["N_TR"]))
        dphi_dt_oil = -self._params["kappa"] * np.sqrt(max(master_s / max(s, 1e-12), 0)) * np.sin(
            phi - master_phi - (self._params["detuning"] * idx * self._dt)
        )
        dphi_dt = dphi_dt_free + dphi_dt_oil
        return dn_dt, ds_dt, dphi_dt

    def step(self) -> None:
        if self._i == len(self._curr):
            raise ValueError("Stepped too far.")

        i_val = self._curr[self._i]
        master_n, master_s, master_phi = self._primary.get()
        n, s, phi = self._n, self._s, self._phi
        dt = self._dt
        idx = self._i

        # --- RK4 integration ---
        k1 = self._derivatives(n, s, phi, i_val, master_n, master_s, master_phi, idx)
        k2 = self._derivatives(n + 0.5 * dt * k1[0], s + 0.5 * dt * k1[1], phi + 0.5 * dt * k1[2], i_val, master_n, master_s, master_phi, idx)
        k3 = self._derivatives(n + 0.5 * dt * k2[0], s + 0.5 * dt * k2[1], phi + 0.5 * dt * k2[2], i_val, master_n, master_s, master_phi, idx)
        k4 = self._derivatives(n + dt * k3[0], s + dt * k3[1], phi + dt * k3[2], i_val, master_n, master_s, master_phi, idx)

        self._n = n + (dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        self._s = max(s + (dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]), 1)
        self._phi = phi + (dt / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])

        # Langevin noise
        super()._langevin_noise()

        # advance primary as well
        self._primary.step()
        self._i += 1
        
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
    return np.where((shifted >= 0) & (phase < duration), peak, bias)

def power(s: float) -> float:
    return ((PARAM["eta"]*h*c)/(PARAM["lambda"]*PARAM["tau_p"]))*s
    

def main() -> None:
    # --- primary (driving laser) current: pulse train + bias ---
    I_primary = make_pulse_train(
        4400 * PARAM["DT"],
        2000 * PARAM["DT"],
        PEAK,
        BIAS,
        1200 * PARAM["DT"]
    )

    # --- secondary (driven laser) current: only bias current ---
    I_secondary = np.full_like(I_primary, BIAS)  # constant bias
    
    # --- propagation objects ---
    primary = StandardPropagation(
        (PARAM["N0"], PARAM["S0"], 0),
        PARAM["DT"], PARAM, I_primary
    )

    secondary = SecondaryPropagation(
        primary,
        (PARAM["N0"], PARAM["S0"], 0),
        PARAM["DT"], PARAM, I_secondary
    )

    # --- histories ---
    n_hist_prim, s_hist_prim, I_hist_prim, phase_hist_prim = [], [], [], []
    n_hist_sec, s_hist_sec, phase_hist_sec = [], [], []
    sum_s_prim,sum_s_sec=0,0
    for j in range(SIM):
        # advance both systems
        secondary.step()

        n_p, s_p, phi_p = primary.get()
        n_s, s_s, phi_s = secondary.get()

        n_hist_prim.append(n_p)
        s_hist_prim.append(s_p)
        sum_s_prim+=s_p
        n_hist_sec.append(n_s)
        s_hist_sec.append(s_s)
        sum_s_sec+=s_s
        I_hist_prim.append(I_primary[j])
        phase_hist_prim.append(phi_p)
        phase_hist_sec.append(phi_s)
    
    peak_power_prim=power(max(s_hist_prim))*1e3
    peak_power_sec=power(max(s_hist_sec))*1e3
    avg_power_prim=power(sum_s_prim/SIM)*1e3
    avg_power_sec=power(sum_s_sec/SIM)*1e3

    t = np.linspace(0, PARAM["DT"] * SIM, SIM)
    # --- plotting ---
    import matplotlib as mpl
    from matplotlib.collections import LineCollection

    def plot_with_phase(t, y, phi, label, cmap="viridis", linewidth=2):
        """
        Plot y(t) with color representing phi(t) using a LineCollection.
        """
        # Create segments for line collection
        points = np.array([t, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Normalize phase to [0,1] for colormap
        norm = mpl.colors.Normalize(vmin=0, vmax=2*np.pi)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth)
        lc.set_array(phi)
        
        ax = plt.gca()
        ax.add_collection(lc)
        ax.autoscale()
        #plt.colorbar(lc, label="Phase φ")
        plt.xlabel("Time (s)")
        plt.ylabel(label)

    # --- plotting histories with phase coloring ---
    plt.figure(figsize=(10, 8))
    phase_hist_prim=np.unwrap(phase_hist_prim)
    phase_hist_sec=np.unwrap(phase_hist_sec)
    plt.subplot(3, 1, 1)
    plot_with_phase(t, n_hist_prim, phase_hist_prim, "","hsv")
    plot_with_phase(t, n_hist_sec, phase_hist_sec, r"Primary/Secondary Carrier Density $N(t)$","twilight_shifted")

    plt.subplot(3, 1, 2)
    plot_with_phase(t, s_hist_prim, phase_hist_prim, "","hsv")
    plot_with_phase(t, s_hist_sec, phase_hist_sec, r"Primary/Secondary Photon Density $S(t)$","twilight_shifted")

    plt.subplot(3, 1, 3)
    plt.plot(t, I_hist_prim, label="Primary I", color="black")
    plt.plot(t, I_secondary, label="Secondary I", color="gray", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Current")
    plt.legend()
    plt.figtext(
        0.5, 0.01,  # normalized coordinates: center horizontally, bottom vertically
        f"Primary Laser: Peak P = {peak_power_prim:.3f}mW, Avg P = {avg_power_prim:.3f}mW | "
        f"Secondary Laser: Peak P = {peak_power_sec:.3f}mW, Avg P = {avg_power_sec:.3f}mW",
        ha="center",
        fontsize=12
    )
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()