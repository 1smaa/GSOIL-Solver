import numpy as np
import json
import os
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

PATH = "parameters.json"

with open(PATH, mode="r", encoding="utf-8") as f:
    PARAM = json.load(f)

SIM = int(1e4)
DT = PARAM["DT"]
T_MAX = SIM * DT

# Define input injection current as a function of time
def I(t):
    # Example rectangular pulse from 2 ns to 10 ns (adjust as needed)
    if 2e-9 <= t <= 10e-9:
        return 3e-6
    else:
        return 0.0

# Vectorize I for faster computation
vecI = np.vectorize(I)

# Laser rate equations as ODE
def laser_rate_eq(t, y):
    N, S, phi = y
    q = PARAM["q"]
    V = PARAM["V"]
    tau_n = PARAM["tau_n"]
    tau_p = PARAM["tau_p"]
    GAMMA = PARAM["GAMMA"]
    g = PARAM["g"]
    N_TR = PARAM["N_TR"]
    e = PARAM["e"]
    beta = PARAM["beta"]
    alpha = PARAM["alpha"]

    # Injection current at time t
    curr = I(t)

    dN_dt = (curr / (q * V)) - (N / tau_n) - (g * (N - N_TR) * S) / (1 + e * S)
    dS_dt = (GAMMA * g * (N - N_TR) * S) / (1 + e * S) - (S / tau_p) + (GAMMA * beta * N / tau_n)
    dphi_dt = 0.5 * alpha * (GAMMA * g * (N - N_TR) - (1 / tau_p))

    return [dN_dt, dS_dt, dphi_dt]

def main():
    y0 = [PARAM["N0"], PARAM["S0"], 0.0]  # Initial conditions: N, S, phi
    t_span = (0, T_MAX)
    t_eval = np.linspace(t_span[0], t_span[1], SIM)

    # Solve ODE with solve_ivp using a stiff solver method (Radau)
    sol = solve_ivp(laser_rate_eq, t_span, y0, t_eval=t_eval, method='Radau')

    N = sol.y[0]
    S = sol.y[1]
    phi = sol.y[2]

    I_arr = vecI(t_eval)

    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t_eval, N)
    plt.title('Carrier Density N')
    plt.xlabel('Time (s)')
    plt.ylabel('N (m^-3)')

    plt.subplot(3, 1, 2)
    plt.plot(t_eval, S)
    plt.title('Photon Density S')
    plt.xlabel('Time (s)')
    plt.ylabel('S (a.u.)')

    plt.subplot(3, 1, 3)
    plt.plot(t_eval, I_arr)
    plt.title('Injection Current I')
    plt.xlabel('Time (s)')
    plt.ylabel('I (A)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
