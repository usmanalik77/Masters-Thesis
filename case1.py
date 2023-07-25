import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from scipy.integrate import odeint

# Simulation parameters
duration = 101  # Duration of the simulation in hours
time_step = 1  # Time step size in hours

# Building parameters
layers = 4  # Number of layers
area_room = 3.65 * 4  # Area of the building in square meters
height_room = 3.3

cp = np.array([835, 1317, 835, 835])  # Specific heat capacity of each layer

# Constants and parameters
rho_in = 1.2  # Air density at inlet, kg/m^3
rho_out = 1.2  # Air density at outlet, kg/m^3
alpha_w = [1.7e-7, 1.4e-6, 2.25e-2, 1.7e-7]  # Thermal diffusivity of wall, m^2/s
alpha_r = [1.7e-7, 1.4e-6, 1.8e-7, 4e-7]  # Thermal diffusivity of roof, m^2/s
alpha_f = [4.0e-7, 1.7e-7]  # Thermal diffusivity of floor, m^2/s
alpha_fur = 1.8e-7  # Thermal diffusivity of furniture, m^2/s
l_w = [6e-3, 75e-3, 8.5e-3, 8.5e-3]  # Thickness of wall layers, m
l_r = [6e-3, 100e-3, 11e-3, 6.5e-4]  # Thickness of roof layers, m
l_f = [1e-3, 9e-3]  # Thickness of floor layers, m
K_w = [0.14, 0.038, 0.026, 0.14]
K_r = [0.14, 0.038, 0.12, 0.027]
K_f = [0.027, 0.14]
K_fur = 2
rho_f = [55.0, 615.0]  # Furniture density, kg/m^3
c_p_f = [1210.0, 1317.0]  # Furniture specific heat capacity, J/(kg*K)

# Outside temperature
T_outside = 270.95  # Outside temperature in degrees kelvin


# Furniture parameters
furniture_volume = 1  # Volume of furniture in cubic meters
furniture_alpha = 1.8e-7  # Thermal diffusivity of furniture
furniture_radius = 0.5  # Radius of furniture in meters

# Heat transfer coefficients
h_walls = 2
h_furniture = 2

air_flow = 0.7

num_steps = int(duration / time_step)
T = np.zeros((num_steps, layers))
# Set initial conditions
V_b = area_room * height_room
V_out = air_flow
# Assume V_in = V_out
V_in = V_out
Ventilation_rate = V_in / V_b
# Assume ρ_in = ρ_out
# rho_in = rho_out

h_b_w = 2
h_alpha_w = 1
h_b_r = 2
h_alpha_r = 1
h_b_f = 2
h_b_fur = 2

h_in = h_b_w + h_alpha_w
h_out = h_b_r + h_alpha_r
R = 8.314  # Pa*m^3/(mol*K)
M_b = 0.02897  # kg/mol
# T_b_occupied_initial = 293.15 # kelvin
# T_b_unoccupied_initial = 288.15 # kelvin
T_b_occupied_initial = 293.15
T_b_unoccupied_initial = 288.15

time = np.arange(0, duration, time_step)
T_b = np.zeros(num_steps)  # Building unit temperature
T_w_occupied = np.zeros((len(l_w), num_steps))  # Wall temperatures
T_r_occupied = np.zeros((len(l_r), num_steps))  # Roof temperatures
T_f_occupied = np.zeros((len(l_f), num_steps))  # Floor temperatures
T_fur_occupied = np.zeros(num_steps)  # Furniture temperature

T_w_unoccupied = np.zeros((len(l_w), num_steps))  # Wall temperatures
T_r_unoccupied = np.zeros((len(l_r), num_steps))  # Roof temperatures
T_f_unoccupied = np.zeros((len(l_f), num_steps))  # Floor temperatures
T_fur_unoccupied = np.zeros(num_steps)  # Furniture temperature

T_b_occupied = np.zeros(num_steps)
T_b_unoccupied = np.zeros(num_steps)
T_b_occupied[0] = T_b_occupied_initial
T_b_unoccupied[0] = T_b_unoccupied_initial
P_atm = 101300  # Pa

# Assume P_b = P_atm
P_b = P_atm
P_in = P_b
rho_b_occupied = (P_b * M_b) / (R * T_b_occupied[0])
rho_b_unoccupied = (P_b * M_b) / (R * T_b_unoccupied[0])

c_p_b = 1005  # J/(kgK)


# Energy
dE_in_dt_occupied = (
    rho_b_occupied * c_p_b * (T_outside - T_b_occupied[0]) * (area_room * h_in)
)
dE_out_dt_occupied = (
    rho_b_occupied * c_p_b * (T_b_occupied[0] - T_outside) * (area_room * h_out)
)

dE_in_dt_unoccupied = (
    rho_b_unoccupied * c_p_b * (T_outside - T_b_unoccupied[0]) * (area_room * h_in)
)
dE_out_dt_unoccupied = (
    rho_b_unoccupied * c_p_b * (T_b_unoccupied[0] - T_outside) * (area_room * h_out)
)

A_window = A_door = A_walls = A_floor = A_roof = area_room  # it's an approximation
U_window = U_door = U_walls = U_floor = U_roof = 1.2  # W/(m^2·K) approximation

dQ_dt_loss_occupied = np.zeros(num_steps)
dQ_dt_loss_unoccupied = np.zeros(num_steps)
dT_b_dt_occupied = np.zeros(num_steps)
dT_b_dt_unoccupied = np.zeros(num_steps)
dQ_dt_supply = np.zeros(num_steps)
T_infinity = T_outside
Insolation = 0  # W/m^2
Appliances = 0  # W
dQ_people_dt = 58.2 * area_room  # W
dQ_people_dt_series = np.zeros(num_steps)
dQ_appliances_dt = Appliances  # W
dQ_solar_dt = Insolation * area_room
Q_heater = 1000  # W]
dQ_heater_dt = np.zeros(num_steps)
# dQ_heater_dt = Q_heater
dQ_dt_supply = np.zeros(num_steps)
k1_occupied = np.zeros(num_steps)
k1_unoccupied = np.zeros(num_steps)
k2_occupied = np.zeros(num_steps)
k2_unoccupied = np.zeros(num_steps)


for i in range(1, num_steps):
    dQ_window_dt_occupied = U_window * A_window * (T_infinity)
    dQ_door_dt_occupied = U_door * A_door * (T_infinity)
    dQ_walls_dt_occupied = U_walls * A_walls * (T_infinity)
    dQ_floor_dt_occupied = U_floor * A_floor * (T_infinity)
    dQ_roof_dt_occupied = U_roof * A_roof * (T_infinity)

    dQ_window_dt_unoccupied = U_window * A_window * (T_infinity)
    dQ_door_dt_unoccupied = U_door * A_door * (T_infinity)
    dQ_walls_dt_unoccupied = U_walls * A_walls * (T_infinity)
    dQ_floor_dt_unoccupied = U_floor * A_floor * (T_infinity)
    dQ_roof_dt_unoccupied = U_roof * A_roof * (T_infinity)

    dQ_dt_loss_occupied = (
        dQ_window_dt_occupied
        + dQ_door_dt_occupied
        + dQ_walls_dt_occupied
        + dQ_floor_dt_occupied
        + dQ_roof_dt_occupied
    )
    dQ_dt_loss_unoccupied = (
        dQ_window_dt_unoccupied
        + dQ_door_dt_unoccupied
        + dQ_walls_dt_unoccupied
        + dQ_floor_dt_unoccupied
        + dQ_roof_dt_unoccupied
    )

    rho_in_occupied = (P_in * M_b) / (R * T_b_occupied[0])
    rho_in_unoccupied = (P_in * M_b) / (R * T_b_unoccupied[0])
    rho_out_occupied = rho_b_occupied
    rho_out_unoccupied = rho_b_unoccupied

    drho_b_occupied_dt = 0
    drho_b_unoccupied_dt = 0

    V_in = V_out = 0.7

    h_in = 60179.47996
    h_out = 60587.96
    dQ_dt_supply[i] = 0
    k1_occupied_0 = 70

    k1_unoccupied_0 = 70
    k1_occupied[i] = (
        (V_in * rho_in_occupied * h_in - V_out * rho_out_occupied * h_out)
        + dQ_dt_supply[i]
        + dQ_dt_loss_occupied
    ) / (V_b * rho_b_occupied * (c_p_b - R / M_b))
    k1_unoccupied[i] = (
        (V_in * rho_in_unoccupied * h_in - V_out * rho_out_unoccupied * h_out)
        + dQ_dt_supply[i]
        + dQ_dt_loss_unoccupied
    ) / (V_b * rho_b_unoccupied * (c_p_b - R / M_b))

    k1_unocc = 70

    k2_occupied[i] = dQ_dt_loss_occupied / T_infinity + (
        V_in * rho_in_occupied - V_out * rho_out_occupied
    ) / (rho_b_occupied * V_b)
    k2_unoccupied[i] = dQ_dt_loss_unoccupied / T_infinity + (
        V_in * rho_in_occupied - V_out * rho_out_occupied
    ) / (rho_b_unoccupied * V_b)
    k2 = 0.23
    T_b_unoccupied[i] = (1 / k2) * (
        k1_unoccupied_0 - (k1_unocc - k2 * T_b_unoccupied[0]) * math.exp(-k2 * (i))
    )

    # Using K[i] variable instead
    #  T_b_unoccupied[i] = (1 / k2_unoccupied[i]) * (
    #     k1_unoccupied[i]
    #     - (k1_unoccupied[i] - k2_unoccupied[i] * T_b_unoccupied[0])
    #     * math.exp(-k2_unoccupied[i] * (i))
    # )

for i in range(len(l_w)):
    T_w_occupied[i][0] = T_infinity
for i in range(len(l_r)):
    T_r_occupied[i][0] = T_infinity
for i in range(len(l_f)):
    T_f_occupied[i][0] = T_infinity
T_fur_occupied[0] = T_b_occupied[0]

for i in range(len(l_w)):
    T_w_unoccupied[i][0] = T_infinity
for i in range(len(l_r)):
    T_r_unoccupied[i][0] = T_infinity
for i in range(len(l_f)):
    T_f_unoccupied[i][0] = T_infinity
T_fur_unoccupied[0] = T_b_unoccupied[0]


firstPart = np.zeros(num_steps)
for step in range(1, num_steps):
    dT_w_dt_unoccupied = np.zeros(len(l_w))
    for i in range(len(l_w)):
        if i == 0:
            dT_w_dt_unoccupied[i] = alpha_w[i] * (
                (
                    T_w_unoccupied[i + 1][step - 1]
                    - 2 * T_w_unoccupied[i][step - 1]
                    + T_b_unoccupied[step - 1]
                )
                / (l_w[i] ** 2)
            )
        elif i == len(l_w) - 1:
            dT_w_dt_unoccupied[i] = alpha_w[i] * (
                (
                    T_w_unoccupied[i - 1][step - 1]
                    - 2 * T_w_unoccupied[i][step - 1]
                    + T_w_unoccupied[i][step - 1]
                )
                / (l_w[i] ** 2)
            )
        else:
            dT_w_dt_unoccupied[i] = alpha_w[i] * (
                (
                    T_w_unoccupied[i + 1][step - 1]
                    - 2 * T_w_unoccupied[i][step - 1]
                    + T_w_unoccupied[i][step - 1]
                )
                / (l_w[i] ** 2)
            )

    dT_r_dt_unoccupied = np.zeros(len(l_r))
    for i in range(len(l_r)):
        if i == 0:
            dT_r_dt_unoccupied[i] = alpha_r[i] * (
                (
                    T_r_unoccupied[i + 1][step - 1]
                    - 2 * T_r_unoccupied[i][step - 1]
                    + T_b_unoccupied[step - 1]
                )
                / (l_r[i] ** 2)
            )
        elif i == len(l_r) - 1:
            dT_r_dt_unoccupied[i] = alpha_r[i] * (
                (
                    T_r_unoccupied[i - 1][step - 1]
                    - 2 * T_r_unoccupied[i][step - 1]
                    + T_r_unoccupied[i][step - 1]
                )
                / (l_r[i] ** 2)
            )
        else:
            dT_r_dt_unoccupied[i] = alpha_r[i] * (
                (
                    T_r_unoccupied[i + 1][step - 1]
                    - 2 * T_r_unoccupied[i][step - 1]
                    + T_r_unoccupied[i][step - 1]
                )
                / (l_r[i] ** 2)
            )

    dT_f_dt_unoccupied = np.zeros(len(l_f))
    for i in range(len(l_f)):
        if i == 0:
            dT_f_dt_unoccupied[i] = alpha_f[i] * (
                (
                    T_f_unoccupied[i + 1][step - 1]
                    - 2 * T_f_unoccupied[i][step - 1]
                    + T_b_unoccupied[step - 1]
                )
                / (l_f[i] ** 2)
            ) + (dQ_dt_supply[step] - dQ_dt_loss_unoccupied) / (rho_f[i] * c_p_f[i])
        elif i == len(l_f) - 1:
            dT_f_dt_unoccupied[i] = alpha_f[i] * (
                (
                    T_f_unoccupied[i - 1][step - 1]
                    - 2 * T_f_unoccupied[i][step - 1]
                    + T_f_unoccupied[i][step - 1]
                )
                / (l_f[i] ** 2)
            ) + (dQ_dt_supply[step] - dQ_dt_loss_unoccupied) / (rho_f[i] * c_p_f[i])
        else:
            dT_f_dt_unoccupied[i] = alpha_f[i] * (
                (
                    T_f_unoccupied[i + 1][step - 1]
                    - 2 * T_f_unoccupied[i][step - 1]
                    + T_f_unoccupied[i][step - 1]
                )
                / (l_f[i] ** 2)
            ) + (dQ_dt_supply[step] - dQ_dt_loss_unoccupied) / (rho_f[i] * c_p_f[i])

    dT_fur_dt_unoccupied = (
        alpha_fur
        * (
            (
                T_fur_occupied[0]
                - 2 * T_fur_unoccupied[step - 1]
                + T_b_unoccupied[step - 1]
            )
            / (0.5**2)
        )
    ) + (alpha_fur / 0.5) * ((T_fur_occupied[0] - T_b_unoccupied[step - 1]) / (1))

    for i in range(len(l_w)):
        T_w_unoccupied[i][step] = (
            T_w_unoccupied[i][step - 1] + time_step * dT_w_dt_unoccupied[i]
        )
    for i in range(len(l_r)):
        T_r_unoccupied[i][step] = (
            T_r_unoccupied[i][step - 1] + time_step * dT_r_dt_unoccupied[i]
        )
    for i in range(len(l_f)):
        T_f_unoccupied[i][step] = (
            T_f_unoccupied[i][step - 1] + time_step * dT_f_dt_unoccupied[i]
        )
    T_fur_unoccupied[step] = (
        T_fur_unoccupied[step - 1] + time_step * dT_fur_dt_unoccupied
    )


def dydt(y0, time, T_outside):
    for i in range(1, num_steps):
        deltaRhoB = (V_in * rho_in - V_out * rho_b_unoccupied) / V_b
        firstPart[i] = (
            V_in * rho_in * h_in
            - V_out * rho_b_unoccupied * h_out
            + dQ_dt_supply[i]
            - dQ_dt_loss_unoccupied
        ) / (V_b * rho_b_unoccupied * (c_p_b - (R / M_b))) - (
            T_b_unoccupied[i] / rho_b_unoccupied
        ) * deltaRhoB
    return firstPart[i]


K = 273.15
y0 = 15 + K
tolerance = 1e-5

sol = odeint(dydt, y0, time, args=(T_outside,), rtol=tolerance, atol=tolerance)


plt.figure(figsize=(12, 6))
ax = plt.subplot(111)
ax.plot(time, T_b_unoccupied, label="T building unoccupied")
for i in range(len(l_w)):
    ax.plot(time, T_w_unoccupied[i], label=f"Wall Layer {i+1} unoccupied")
for i in range(len(l_r)):
    ax.plot(time, T_r_unoccupied[i], label=f"Roof Layer {i+1} unoccupied")
for i in range(len(l_f)):
    ax.plot(time, T_f_unoccupied[i], label=f"Floor Layer {i+1} unoccupied")
ax.plot(time, T_fur_unoccupied, label="Furniture unoccupied")
plt.xlabel("Time (hours)")
plt.ylabel("Temperature (°K)")
plt.title("Temperature Variations of Internal Elements - Unoccupied")

ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", ncol=1, borderaxespad=0.0)

plt.grid(True)
plt.tight_layout()
plt.show()
