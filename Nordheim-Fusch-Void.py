import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# -------------------------
# Constants and Parameters
# -------------------------

# Reactor physics
rho_0 = 0.024  # initial reactivity (unitless)
alpha_T = 1.2e-5  # Doppler coefficient [1/K]
alpha_void = 0.025
ell = 0.01  # prompt neutron lifetime [s]

# Fuel properties
m0 = 190000  # fuel mass [kg]
c_solid =  300  # heat capacity of solid Uranium Dioxide [J/kg/K]
c_liquid = 506 # heat capacity of liquid Uranium Dioxide [J/kg/K]
T_melt = 3120 # melting point of uranium dioxide
L_f = 259230 # lateent heat of fusion of uranium dioxide

# Coolant parameters
H = 7  # core height [m]
mdot = 10417  # coolant mass flow rate [kg/s]
z_sat = 2  # height where boiling starts [m]
h_l_sat = 1267  # saturated liquid enthalpy [kJ/kg]
h_v_sat = 2772  # saturated vapor enthalpy [kJ/kg]
rho_l_sat = 739  # saturated liquid density [kg/m^3]
tau_l = 1.35  # specific volume of liquid [L/kg]
tau_v = 27.8  # specific volume of vapor [L/kg]

# -------------------------
# Void Reactivity Function
# -------------------------

def compute_void_reactivity(P):
    # Step 1: Enthalpy gradient [kJ/kg/m]
    delta_h = (P * 1000) / (H * mdot)

    # Step 2: Inlet and outlet enthalpy [kJ/kg]
    h_in = h_l_sat - z_sat * delta_h
    h_out = h_l_sat + (H - z_sat) * delta_h

    # Step 3: Outlet steam quality (void fraction)
    x_out = (h_out - h_l_sat) / (h_v_sat - h_l_sat)
    x_out = np.clip(x_out, 0, 1)  # keep in physical bounds

    # Step 4: Outlet specific volume and density
    tau_out = (1 - x_out) * tau_l + x_out * tau_v  # [L/kg]
    rho_out = 1000 / tau_out  # [kg/m^3]

    # Step 5: Average density
    rho_ave = (1 / H) * (rho_l_sat * z_sat + 0.5 * (rho_l_sat + rho_out) * (H - z_sat))

    # Step 6: Coolant reactivity in pcm
    rho_coolant_pcm = alpha_void * (rho_l_sat - rho_ave) / rho_l_sat
    return rho_coolant_pcm

# -------------------------
# Differential Equations
# -------------------------

def get_fuel_temperature(Q):
    Q_melt_start = m0 * c_solid * T_melt
    Q_melt_end = Q_melt_start + m0 * L_f

    if Q < Q_melt_start:
        return Q / (m0 * c_solid)
    elif Q < Q_melt_end:
        return T_melt
    else:
        Q_liquid = Q - Q_melt_end
        return T_melt + Q_liquid / (m0 * c_liquid)


def reactor_kinetics(t, y):
    P, Q = y
    T = get_fuel_temperature(Q)  # [K]
    rho_coolant = compute_void_reactivity(P)
    rho_total = rho_0 - alpha_T * T + rho_coolant
    dPdt = (rho_total / ell) * P
    dQdt = P * 1e6  # convert MW to W (i.e., J/s)
    return [dPdt, dQdt]

# -------------------------
# Solve the System
# -------------------------

# Initial conditions
P0 = 1000  # MW
Q0 = 0  # MJ

t_span = (0, 10)  # seconds
y0 = [P0, Q0]

solution = solve_ivp(reactor_kinetics, t_span, y0, method='RK45', dense_output=True, max_step=0.01)

# Extract results
t = solution.t
P = solution.y[0]
Q = solution.y[1]
T_fuel = np.array([get_fuel_temperature(q) for q in Q])  # [K]


# --- Peak Power and Total Energy output calculation --- #
P_peak_numerical = np.max(P) #Peak power
t_peak_numerical = t[np.argmax(P)] #Time power peaks
E_total_numerical = Q[-1] / 1e9  #  Total energy released in GJ
print(P_peak_numerical, t_peak_numerical, E_total_numerical)

# -------------------------
# Plot Results
# -------------------------

# Annotation text
annotation_text = (
    f"Peak Power: {P_peak_numerical:,.0f} MW at t = {t_peak_numerical:.2f} s\n"
    f"Total Energy Released: {E_total_numerical :.2f} GJ"
)


plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.title("Power Excursion vs. Time")
plt.plot(t, P, 'k',label='Power (MW)')
plt.ylabel('Power [MW]')
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.title("Fuel Temperature of RBMK reactor vs. Time")
plt.plot(t, T_fuel, 'r-',label='Fuel Temperature [K]')
plt.xlabel('Time [s]')
plt.ylabel('Temperature [K]')
plt.grid()
plt.legend()
plt.text(
    0.05, 0.95, annotation_text,
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.7)
)
plt.tight_layout()
plt.show()
