# --- Modified Reactor Model with Lumped Coolant Heat Transfer and Modulated h --- #

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# -------------------------
# Constants and Parameters
# -------------------------

# Reactor physics
rho_0 = 0.024
alpha_T = 1.2e-5
alpha_void = 0.025
ell = 0.01

# Fuel properties
m0 = 190000  # kg
c_solid = 300  # J/kg/K
c_liquid = 506  # J/kg/K
c_vapor = 900  # J/kg/K (estimated for UO2 vapor)
T_melt = 3120  # K
T_boil = 3815  # K
L_f = 259230  # J/kg
L_v = 1588122  # J/kg
A_ht = 1000  # m^2, effective heat transfer area

# Coolant properties
cp_coolant = 4200  # J/kg/K
m_coolant = 10000  # kg (effective lumped coolant mass)
T_coolant_0 = 550  # K (typical RBMK outlet temp)
h_liquid = 10000  # W/m^2/K
h_vapor = 200     # W/m^2/K

# Core thermo-hydraulics
H = 7
mdot = 10417
z_sat = 2
h_l_sat = 1267
h_v_sat = 2772
rho_l_sat = 739
tau_l = 1.35
tau_v = 27.8

# -------------------------
# Void Reactivity Function
# -------------------------

def compute_void_reactivity(P):
    delta_h = (P * 1000) / (H * mdot)
    h_in = h_l_sat - z_sat * delta_h
    h_out = h_l_sat + (H - z_sat) * delta_h
    x_out = (h_out - h_l_sat) / (h_v_sat - h_l_sat)
    x_out = np.clip(x_out, 0, 1)
    tau_out = (1 - x_out) * tau_l + x_out * tau_v
    rho_out = 1000 / tau_out
    rho_ave = (1 / H) * (rho_l_sat * z_sat + 0.5 * (rho_l_sat + rho_out) * (H - z_sat))
    rho_coolant_pcm = alpha_void * (rho_l_sat - rho_ave) / rho_l_sat
    return rho_coolant_pcm, x_out

# -------------------------
# Fuel Temperature Function
# -------------------------

def get_fuel_temperature(Q):
    Q_melt_start = m0 * c_solid * T_melt
    Q_melt_end = Q_melt_start + m0 * L_f
    Q_vap_start = Q_melt_end + m0 * c_liquid * (T_boil - T_melt)
    Q_vap_end = Q_vap_start + m0 * L_v

    if Q < Q_melt_start:
        return Q / (m0 * c_solid)
    elif Q < Q_melt_end:
        return T_melt
    elif Q < Q_vap_start:
        Q_liquid = Q - Q_melt_end
        return T_melt + Q_liquid / (m0 * c_liquid)
    elif Q < Q_vap_end:
        return T_boil
    else:
        Q_superheat = Q - Q_vap_end
        return T_boil + Q_superheat / (m0 * c_vapor)

# -------------------------
# Reactor Dynamics Function
# -------------------------

def reactor_kinetics(t, y):
    P, Q_fuel, T_coolant = y
    T_fuel = get_fuel_temperature(Q_fuel)

    rho_coolant, x_out = compute_void_reactivity(P)
    rho_total = rho_0 - alpha_T * T_fuel + rho_coolant
    dPdt = (rho_total / ell) * P

    h_eff = h_liquid * (1 - x_out) + h_vapor * x_out
    Q_transfer = h_eff * A_ht * (T_fuel - T_coolant)

    dQ_fuel_dt = P * 1e6 - Q_transfer
    dT_coolant_dt = Q_transfer / (m_coolant * cp_coolant)

    return [dPdt, dQ_fuel_dt, dT_coolant_dt]

# -------------------------
# Initial Conditions & Solve
# -------------------------

P0 = 200
Q0 = 0
T_c0 = T_coolant_0
y0 = [P0, Q0, T_c0]
t_span = (0, 10)

solution = solve_ivp(reactor_kinetics, t_span, y0, method='RK45', dense_output=True, max_step=0.01)

# Extract results
t = solution.t
P = solution.y[0]
Q_fuel = solution.y[1]
T_fuel = np.array([get_fuel_temperature(q) for q in Q_fuel])
T_coolant = solution.y[2]

# Results
P_peak = np.max(P)
t_peak = t[np.argmax(P)]
E_total = Q_fuel[-1] / 1e9
T_fuel_peak = np.max(T_fuel)
t_fuel_peak = t[np.argmax(T_fuel)]
T_coolant_peak = np.max(T_coolant)
t_coolant_peak = t[np.argmax(T_coolant)]

# Plot
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, P, 'k', label='Power (MW)')
plt.ylabel('Power [MW]')
plt.grid()
plt.legend()
annotation = (
    rf"$P_{{\max}}$ = {P_peak:,.0f} MW"+"\n"
    rf"$t_{{\max}}$ = {t_peak:.2f} s"+"\n"
    rf"$E_{{\mathrm{{tot}}}}$ = {E_total:.2f} GJ"
)
plt.text(0.01, 0.95, annotation, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

plt.subplot(3, 1, 2)
plt.plot(t, T_fuel, 'r', label='Fuel Temp [K]')
plt.ylabel('Fuel Temp [K]')
plt.grid()
plt.legend()
plt.plot(t_fuel_peak, T_fuel_peak, 'ro')
plt.text(t_fuel_peak, T_fuel_peak + 200, f'Peak: {T_fuel_peak:.0f} K', color='r')

plt.subplot(3, 1, 3)
plt.plot(t, T_coolant, 'b', label='Coolant Temp [K]')
plt.xlabel('Time [s]')
plt.ylabel('Coolant Temp [K]')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
