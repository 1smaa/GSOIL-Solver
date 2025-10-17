
import sys
import numpy as np
import json
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from scipy.constants import h, c
from scipy.fft import fft
import matplotlib.ticker as mticker
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
# --- Load parameters ---
PATH="phase_seeding.json"
with open(PATH,mode="r",encoding="utf-8") as f:
    PARAM=json.load(f)
SIM=int(10e4)

# --- Example laser parameters ---
BIAS_1, PEAK_1 = 15e-3, 28.9e-3
BIAS_2, PEAK_2 = 0,0
PERIOD_1, PULSE_1 = 30000,15000
PERIOD_2, PULSE_2 = 5000, 2500
REFERENCE_PULSE_1, REFERENCE_PULSE_2 = 0, 7

# --- PyQt6 GUI ---
from matplotlib.colors import Normalize

class LaserPlotter(QMainWindow):
    def __init__(self, t, s_hist_prim, s_hist_sec,
                 phase_hist_prim, phase_hist_sec,
                 n_hist_prim, n_hist_sec,
                 wavelengths, E_fft_prim_shifted, E_fft_sec_shifted,
                 peak_avg_power):
        super().__init__()
        self.setWindowTitle("Laser Simulation Viewer")
        self.domain = "time"

        self._t = t
        self._s_hist_prim = s_hist_prim
        self._s_hist_sec = s_hist_sec
        self._phase_hist_prim = phase_hist_prim
        self._phase_hist_sec = phase_hist_sec
        self._n_hist_prim = n_hist_prim
        self._n_hist_sec = n_hist_sec
        self._wavelengths = wavelengths
        self._E_fft_prim_shifted = E_fft_prim_shifted
        self._E_fft_sec_shifted = E_fft_sec_shifted
        self._peak_avg_power = peak_avg_power

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # --- Figure ---
        self.fig, self.axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1, 1], 'hspace': 0.35})

        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

        self._colorbars = []

        # --- Button ---
        self.button = QPushButton("Switch to Frequency Domain")
        self.button.clicked.connect(self.toggle_domain)
        self.layout.addWidget(self.button)

        # --- Power label ---
        self.power_label = QLabel()
        self.layout.addWidget(self.power_label)
        self.update_power_label()

        self.plot_time_domain()

    def update_power_label(self):
        text = (f"Primary Laser: Peak = {self._peak_avg_power['peak_prim']:.3f} mW | "
                f"Avg = {self._peak_avg_power['avg_prim']:.3f} mW\n"
                f"Secondary Laser: Peak = {self._peak_avg_power['peak_sec']:.3f} mW | "
                f"Avg = {self._peak_avg_power['avg_sec']:.3f} mW")
        self.power_label.setText(text)

    def plot_with_phase(self, ax, y, phi, label, cmap="viridis", cbar_position=0):
        """
        Plot y(t) with color representing phase phi(t) using LineCollection on given axis.
        cbar_position: 0 for top, 1 for bottom, allows vertical stacking of multiple colorbars
        """
        phi = np.mod(phi, 2*np.pi)/np.pi
        points = np.array([self._t, y]).T.reshape(-1,1,2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = Normalize(vmin=phi.min(), vmax=phi.max())
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2)
        lc.set_array(phi)
        ax.add_collection(lc)
        ax.autoscale()

        # --- colorbar vertically stacked ---
        # Use figure coordinates for colorbar: left=0.92, width=0.02, height=0.4, adjust y0 based on cbar_position
        y0 = 0.55 if cbar_position == 0 else 0.1  # top or bottom
        cax = self.fig.add_axes([0.92, y0, 0.02, 0.35])  # [left, bottom, width, height]
        cbar = self.fig.colorbar(lc, cax=cax)
        cbar.set_label(f"Phase φ ({label})")
        self._colorbars.append(cbar)


    def remove_colorbars(self):
        for cb in self._colorbars:
            cb.remove()
        self._colorbars = []

    def plot_time_domain(self):
        self.remove_colorbars()
        for ax in self.axs:
            ax.clear()
            ax.set_visible(True)

        # --- Photon density with phase coloring ---
        self.plot_with_phase(self.axs[0], self._s_hist_prim, self._phase_hist_prim, "Master", "rainbow", cbar_position=0)
        self.plot_with_phase(self.axs[0], self._s_hist_sec, self._phase_hist_sec, "Slave", "twilight_shifted", cbar_position=1)
        self.axs[0].set_ylabel("Photon Density (a.u.)")
        self.axs[0].set_title("Photon Density (Phase Colored)")
        self.axs[0].grid(True)

        # --- Carrier density ---
        self.axs[1].plot(self._t, self._n_hist_prim, label="Master Carrier Density", color="blue")
        self.axs[1].plot(self._t, self._n_hist_sec, label="Slave Carrier Density", color="green")
        self.axs[1].set_xlabel("Time (s)")
        self.axs[1].set_ylabel("Carrier Density (a.u.)")
        self.axs[1].set_title("Carrier Density")
        self.axs[1].legend()
        self.axs[1].grid(True)

        # --- Phase evolution ---
        self.axs[2].plot(self._t, np.mod(self._phase_hist_prim, 2*np.pi)/np.pi, label="Master Phase", color="blue")
        self.axs[2].plot(self._t, np.mod(self._phase_hist_sec, 2*np.pi)/np.pi, label="Slave Phase", color="green")
        self.axs[2].set_xlabel("Time (s)")
        self.axs[2].set_ylabel(r"Phase (rad/$\pi$)")
        self.axs[2].set_title("Phase Evolution")
        self.axs[2].legend()
        self.axs[2].grid(True)
        self.axs[2].minorticks_on()
        self.canvas.draw()

    def plot_frequency_domain(self):
        self.remove_colorbars()
        # hide second subplot
        self.axs[1].set_visible(False)
        self.axs[0].clear()

        self.axs[0].plot(self._wavelengths*1e9, np.abs(self._E_fft_prim_shifted), label="Primary")
        self.axs[0].plot(self._wavelengths*1e9, np.abs(self._E_fft_sec_shifted), label="Secondary")
        self.axs[0].set_xlabel("Wavelength (nm)")
        self.axs[0].set_ylabel("Amplitude (a.u.)")
        self.axs[0].set_title("Frequency Domain")
        self.axs[0].legend()
        self.axs[0].grid(True, which='both', linestyle='--', alpha=0.5)
        self.axs[0].minorticks_on()
        self.axs[0].set_xlim(1555, 1566)
        self.canvas.draw()

    def toggle_domain(self):
        if self.domain == "time":
            self.domain = "frequency"
            self.button.setText("Switch to Time Domain")
            self.plot_frequency_domain()
        else:
            self.domain = "time"
            self.button.setText("Switch to Frequency Domain")
            self.plot_time_domain()



        
        
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
        dphi_dt = (self._params["alpha"] * 0.5 * self._params["g"] * (n - self._params["N_TR"]) - 1/self._params["tau_p"])
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
        self._phi = (phi + (dt / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])) 

        # add Langevin noise
        #self._langevin_noise()
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

        dphi_dt_free = self._params["alpha"] * 0.5 * (self._params["g"] * (n - self._params["N_TR"])) - 1/self._params["tau_p"]
        dphi_dt_oil = -self._params["kappa"] * np.sqrt(max(master_s / max(s, 1e-12), 0)) * np.sin(
            phi - master_phi - (self._params["detuning"] * idx * self._dt)
        )
        dphi_dt = (dphi_dt_free + dphi_dt_oil) 
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
        self._phi = (phi + (dt / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]))

        # Langevin noise
        #super()._langevin_noise()

        # advance primary as well
        self._primary.step()
        self._i += 1
        
    def get(self) -> tuple[float,float,float]:
        return self._n,self._s,self._phi
    
    def get_index(self) -> int:
        return self._i

def phase_seeding(modulation_time: float) -> float:
    return PARAM["seeding"]["shift"]*np.pi*2*PARAM["q"]/(PARAM["alpha"]*PARAM["e"]*modulation_time)
        

def make_pulse_train(period: float, duration: float, peak: float, bias: float, delay: float = 0.0) -> np.ndarray:
    """
    Generate a rectangular pulse train with period, duration, peak, bias, and initial delay.
    """
    dt = PARAM["DT"]
    t = np.arange(SIM) * dt
    shifted = t - delay
    phase = np.mod(shifted, period)
    return np.where((shifted >= 0) & (phase < duration), peak, bias)

def make_slave_train(pulses_per_pulse: int,master_period: float,master_duration: float,peak: float,bias: float) -> np.ndarray:
    fuckass_curr=np.ones(SIM)*bias
    slave_period=master_period/pulses_per_pulse
    slave_peak_time=slave_period/2
    for i in range(SIM):
        j=i%(master_period)
        if j>master_duration: continue
        k=j%slave_period
        if k<slave_peak_time: fuckass_curr[i]=peak
    return fuckass_curr

def make_phase_seeding_master(period: float,duration: float,bias: float,delay: float=0.0) -> np.ndarray:
    dt = PARAM["DT"]
    modulation_time=period-duration
    modulation_i_delta=phase_seeding(modulation_time*dt)
    peak_i=bias+modulation_i_delta
    curr=np.zeros(SIM)
    for i in range(SIM):
        r=i%period
        if r<duration: curr[i]=bias
        else: curr[i]=peak_i
    return curr
    
def power(s: float) -> float:
    return ((PARAM["eta"]*h*c)/(PARAM["lambda"]*PARAM["tau_p"]))*s

def elab_phase(period: int,reference_pulse: int,phase_hist: np.ndarray) -> np.ndarray:
    #for j in range(SIM):
    #    phase_hist[j]=np.mod(phase_hist[j]-phase_hist[(j%period)+(period*reference_pulse)],2*np.pi)
    return phase_hist

def elab_phase_seeding(period: int,reference_pulse: int, phase_hist: np.ndarray) -> np.ndarray:
    j=period
    while j<SIM:
        pass

# --- Run ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    I_primary = make_phase_seeding_master(
        PERIOD_1,
        PULSE_1,
        BIAS_1,
        0
    )
    plt.plot(np.arange(0,SIM,1),I_primary)
    plt.show()
    
    I_secondary=make_slave_train(
        4,
        PERIOD_1,
        PEAK_1,
        PEAK_2,
        BIAS_2
    )
    
    I_secondary=np.zeros(SIM)
    # --- secondary (driven laser) current: only bias current ---
    #I_secondary = make_pulse_train(
    #    PERIOD_2 * PARAM["DT"],
    #    PULSE_2 * PARAM["DT"],
    #    PEAK_2,
    #    BIAS_2,
    #    0
    #)
    
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
    s_hist_prim = np.array(s_hist_prim)
    s_hist_sec = np.array(s_hist_sec)
    phase_hist_prim = np.array(phase_hist_prim)
    phase_hist_sec = np.array(phase_hist_sec)

    # --- compute electric fields ---
    E_primary = np.sqrt(s_hist_prim) * np.exp(1j * phase_hist_prim)
    E_secondary = np.sqrt(s_hist_sec) * np.exp(1j * phase_hist_sec)

    # --- FFT ---
    E_fft_prim = fft(E_primary)
    E_fft_sec = fft(E_secondary)

    # --- frequency axis ---
    freqs = np.fft.fftfreq(len(E_primary), d=PARAM["DT"])
    freqs_shifted = np.fft.fftshift(freqs)
    E_fft_prim_shifted = np.fft.fftshift(E_fft_prim)
    E_fft_sec_shifted = np.fft.fftshift(E_fft_sec)

    # --- center around 1560 nm ---
    lambda0 = 1560e-9  # meters
    f0 = 3e8 / lambda0  # optical carrier in Hz
    optical_freqs = freqs_shifted + f0  # shift FFT around optical carrier
    wavelengths = 3e8 / optical_freqs  # convert to meters
    phase_hist_prim=elab_phase(PERIOD_1,REFERENCE_PULSE_1,phase_hist_prim)
    phase_hist_sec=elab_phase(PERIOD_1,REFERENCE_PULSE_1,phase_hist_sec)
    window=LaserPlotter(
        t,s_hist_prim,s_hist_sec,phase_hist_prim,phase_hist_sec,n_hist_prim,n_hist_sec,wavelengths,E_fft_prim_shifted,E_fft_sec_shifted,{
            "peak_prim":peak_power_prim,
            "peak_sec":peak_power_sec,
            "avg_prim":avg_power_prim,
            "avg_sec":avg_power_sec
        }
    )
    window.show()
    sys.exit(app.exec())
