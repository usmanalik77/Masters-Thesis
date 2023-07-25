import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint

# Simulation parameters
duration = 100  # Duration of the simulation in hours
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
alpha_f = [4.0e-7, 1.7e-7, 1.4e-6]  # Thermal diffusivity of floor, m^2/s
alpha_fur = 1.8e-7  # Thermal diffusivity of furniture, m^2/s
l_w = [6e-3, 75e-3, 8.5e-3, 8.5e-3]  # Thickness of wall layers, m
l_r = [6e-3, 100e-3, 11e-3, 6.5e-4]  # Thickness of roof layers, m
l_f = [1e-3, 9e-3, 75e-3]  # Thickness of floor layers, m
K_w = [0.14, 0.038, 0.026, 0.14]
K_r = [0.14, 0.038, 0.12, 0.027]
K_f = [0.027, 0.14, 0.038]
K_fur = 2
rho_f = [55.0, 615.0, 32.0]  # Furniture density, kg/m^3
c_p_f = [1210.0, 1317.0, 835.0]  # Furniture specific heat capacity, J/(kg*K)


# Furniture parameters
furniture_volume = 1  # Volume of furniture in cubic meters
furniture_alpha = 1.8e-7  # Thermal diffusivity of furniture
furniture_radius = 0.5  # Radius of furniture in meters

# Heat transfer coefficients
h_walls = 2
h_furniture = 2

air_flow = 0.7

num_steps = int(duration / time_step)
T_outside = np.zeros(num_steps)
t = np.linspace(0, 1, num_steps)
for i in range(num_steps):
    T_outside[i] = 2 * (
        np.sin(2 * np.pi * 270 * t[i]) + 270.95 / 2
    )  # Outside temperature in degrees Celsius

T = np.zeros((num_steps, layers))
# Set initial conditions
V_b = area_room * height_room
V_out = air_flow
# Assume V_in = V_out
V_in = V_out
Ventilation_rate = V_in / V_b

h_b_w = 2
h_alpha_w = 1
h_b_r = 2
h_alpha_r = 1
h_b_f = 2
h_b_fur = 2

h_in = h_b_w + h_alpha_w
h_out = h_b_r + h_alpha_r
R = 8.314  # J/(mol*K)
M_b = 0.02897  # g/mol
T_b_occupied_initial = 293.15

time = np.arange(0, duration, time_step)
T_b = np.zeros(num_steps)  # Building unit temperature
T_w_occupied = np.zeros((len(l_w), num_steps))  # Wall temperatures
T_r_occupied = np.zeros((len(l_r), num_steps))  # Roof temperatures
T_f_occupied = np.zeros((len(l_f), num_steps))  # Floor temperatures
T_fur_occupied = np.zeros(num_steps)  # Furniture temperature

T_b_occupied = np.zeros(num_steps)
T_b_occupied[0] = T_b_occupied_initial
P_atm = 101300  # Pa

# Assume P_b = P_atm
P_b = P_atm
P_in = P_b
rho_b_occupied = np.zeros(num_steps)


c_p_b = 1005  # J/(kgK)
dE_in_dt_occupied = np.zeros(num_steps)
dE_out_dt_occupied = np.zeros(num_steps)
dE_in_dt_unoccupied = np.zeros(num_steps)
dE_out_dt_unoccupied = np.zeros(num_steps)
T_infinity = np.zeros(num_steps)

T_infinity[i] = T_outside[i]

A_window = A_door = A_walls = A_floor = A_roof = area_room  # it's an approximation
U_window = U_door = U_walls = U_floor = U_roof = 1.2  # W/(m^2·K) approximation

dQ_dt_loss_occupied = np.zeros(num_steps)
dT_b_dt_occupied = np.zeros(num_steps)
dQ_dt_supply = np.zeros(num_steps)

Insolation = 100  # W/m^2
Appliances = 150  # W
dQ_people_dt = 58.2 * area_room  # W
dQ_people_dt_series = np.zeros(num_steps)
dQ_appliances_dt = Appliances  # W
dQ_solar_dt = Insolation * area_room
Q_heater = 1000  # W]
dQ_heater_dt = np.zeros(num_steps)
dQ_dt_supply = np.zeros(num_steps)
k1_occupied = np.zeros(num_steps)
k2_occupied = np.zeros(num_steps)

dQ_window_dt_occupied = np.zeros(num_steps)
dQ_door_dt_occupied = np.zeros(num_steps)
dQ_walls_dt_occupied = np.zeros(num_steps)
dQ_floor_dt_occupied = np.zeros(num_steps)
dQ_roof_dt_occupied = np.zeros(num_steps)
rho_in_occupied = np.zeros(num_steps)
rho_out_occupied = np.zeros(num_steps)
dQ_dt_loss_occupied = np.zeros(num_steps)
rho_b_occupied[0] = (P_b * M_b) / (R * T_b_occupied[0])
rho_in_occupied[0] = (P_in * M_b) / (R * T_b_occupied[0])
rho_out_occupied[0] = rho_b_occupied[0]

for i in range(1, num_steps):
    dQ_window_dt_occupied[i] = U_window * A_window * (T_infinity[i])
    dQ_door_dt_occupied[i] = U_door * A_door * (T_infinity[i])
    dQ_walls_dt_occupied[i] = U_walls * A_walls * (T_infinity[i])
    dQ_floor_dt_occupied[i] = U_floor * A_floor * (T_infinity[i])
    dQ_roof_dt_occupied[i] = U_roof * A_roof * (T_infinity[i])
    dQ_dt_loss_occupied[i] = (
        dQ_window_dt_occupied[i]
        + dQ_door_dt_occupied[i]
        + dQ_walls_dt_occupied[i]
        + dQ_floor_dt_occupied[i]
        + dQ_roof_dt_occupied[i]
    )

    drho_b_occupied_dt = 0
    drho_b_unoccupied_dt = 0

    if i <= 50:
        dQ_dt_supply[i] = 1000
        V_in = V_out = 0
        # rho_in_occupied = rho_out_occupied = 1.1041
        h_in = 60179.47996
        h_out = 60587.96
        k1_occupied_0 = 90
        k1_occupied[i] = (
            (
                V_in * rho_in_occupied[i - 1] * h_in
                - V_out * rho_out_occupied[i - 1] * h_out
            )
            + dQ_dt_supply[i]
            + dQ_dt_loss_occupied[i]
        ) / (V_b * rho_b_occupied[i - 1] * (c_p_b - R / M_b))
        k1 = 58.7
        k2_occupied[i] = dQ_dt_loss_occupied[i] / T_infinity[i] + (
            V_in * rho_in_occupied[i - 1] - V_out * rho_out_occupied[i - 1]
        ) / (rho_b_occupied[i - 1] * V_b)
        k2 = 0.2
        T_b_occupied[i] = (1 / k2) * (
            k1_occupied_0 - (k1 - k2 * T_b_occupied[0]) * math.exp(-k2 * i)
        )
        # T_b_occupied[i] =  (1/k2) * (k1_occupied_0 - (k1_occupied[i] - k2* T_b_occupied[0])*math.exp(-k2*i))
    else:
        V_in = V_out = 0
        # rho_in_occupied = rho_out_occupied = 1.1041
        h_in = 61173.08
        h_out = 60587.96
        dQ_dt_supply[i] = 0
        k1_occupied_50 = 50
        k1_occupied[i] = (
            (
                V_in * rho_in_occupied[i - 1] * h_in
                - V_out * rho_out_occupied[i - 1] * h_out
            )
            + dQ_dt_supply[i]
            + dQ_dt_loss_occupied[i]
        ) / (V_b * rho_b_occupied[i - 1] * (c_p_b - R / M_b))
        k1 = 90
        k2_occupied[i] = dQ_dt_loss_occupied[i] / T_infinity[i] + (
            V_in * rho_in_occupied[i - 1] - V_out * rho_out_occupied[i - 1]
        ) / (rho_b_occupied[i - 1] * V_b)
        k2 = 0.2

        T_b_occupied[i] = (1 / k2) * (
            k1_occupied_50 - (k1 - k2 * T_b_occupied[50]) * math.exp(-k2 * (i - 50))
        )
        # T_b_occupied[i] =   (1/k2) * (k1_occupied_50 - (k1_occupied[i] - k2* T_b_occupied[50])*math.exp(-k2*(i-50)))
    rho_b_occupied[i] = (P_b * M_b) / (R * T_b_occupied[i])
    rho_in_occupied[i] = (P_in * M_b) / (R * T_b_occupied[i])
    rho_out_occupied[i] = rho_b_occupied[i]
for i in range(len(l_w)):
    T_w_occupied[i][0] = T_infinity[0]
for i in range(len(l_r)):
    T_r_occupied[i][0] = T_infinity[0]
for i in range(len(l_f)):
    T_f_occupied[i][0] = T_infinity[0]
T_fur_occupied[0] = T_b_occupied[0]
firstPart = np.zeros(num_steps)

for step in range(1, num_steps):
    dT_w_dt_occupied = np.zeros(len(l_w))

    for i in range(len(l_w)):
        if i == 0:
            dT_w_dt_occupied[i] = alpha_w[i] * (
                (
                    T_w_occupied[i + 1][step - 1]
                    - 2 * T_w_occupied[i][step - 1]
                    + T_b_occupied[step - 1]
                )
                / (l_w[i] ** 2)
            )
        elif i == len(l_w) - 1:
            dT_w_dt_occupied[i] = alpha_w[i] * (
                (
                    T_w_occupied[i - 1][step - 1]
                    - 2 * T_w_occupied[i][step - 1]
                    + T_w_occupied[i][step - 1]
                )
                / (l_w[i] ** 2)
            )
        else:
            dT_w_dt_occupied[i] = alpha_w[i] * (
                (
                    T_w_occupied[i + 1][step - 1]
                    - 2 * T_w_occupied[i][step - 1]
                    + T_w_occupied[i][step - 1]
                )
                / (l_w[i] ** 2)
            )

    dT_r_dt_occupied = np.zeros(len(l_r))
    for i in range(len(l_r)):
        if i == 0:
            dT_r_dt_occupied[i] = alpha_r[i] * (
                (
                    T_r_occupied[i + 1][step - 1]
                    - 2 * T_r_occupied[i][step - 1]
                    + T_b_occupied[step - 1]
                )
                / (l_r[i] ** 2)
            )
        elif i == len(l_r) - 1:
            dT_r_dt_occupied[i] = alpha_r[i] * (
                (
                    T_r_occupied[i - 1][step - 1]
                    - 2 * T_r_occupied[i][step - 1]
                    + T_r_occupied[i][step - 1]
                )
                / (l_r[i] ** 2)
            )
        else:
            dT_r_dt_occupied[i] = alpha_r[i] * (
                (
                    T_r_occupied[i + 1][step - 1]
                    - 2 * T_r_occupied[i][step - 1]
                    + T_r_occupied[i][step - 1]
                )
                / (l_r[i] ** 2)
            )

    dT_f_dt_occupied = np.zeros(len(l_f))
    for i in range(len(l_f)):
        if i == 0:
            dT_f_dt_occupied[i] = alpha_f[i] * (
                (
                    T_f_occupied[i + 1][step - 1]
                    - 2 * T_f_occupied[i][step - 1]
                    + T_b_occupied[step - 1]
                )
                / (l_f[i] ** 2)
            ) + (dQ_dt_supply[step] - dQ_dt_loss_occupied[i]) / (rho_f[i] * c_p_f[i])
            # print(f"{i+1}", T_f_occupied[i+1][step-1])
        elif i == len(l_f) - 1:
            dT_f_dt_occupied[i] = alpha_f[i] * (
                (
                    T_outside[step - 1]
                    - 2 * T_f_occupied[i][step - 1]
                    + T_f_occupied[i][step - 1]
                )
                / (l_f[i] ** 2)
            ) + (dQ_dt_supply[step] - dQ_dt_loss_occupied[i]) / (rho_f[i] * c_p_f[i])

        else:
            dT_f_dt_occupied[i] = alpha_f[i] * (
                (
                    T_f_occupied[i + 1][step - 1]
                    - 2 * T_f_occupied[i][step - 1]
                    + T_f_occupied[i][step - 1]
                )
                / (l_f[i] ** 2)
            ) + (dQ_dt_supply[step] - dQ_dt_loss_occupied[i]) / (rho_f[i] * c_p_f[i])
            # print(f"{i}", T_f_occupied[i+1][step-1])
    dT_fur_dt_occupied = (
        alpha_fur
        * (
            (T_fur_occupied[0] - 2 * T_fur_occupied[step - 1] + T_b_occupied[step - 1])
            / (0.5**2)
        )
    ) + (alpha_fur / 0.5) * ((T_fur_occupied[0] - T_b_occupied[step - 1]) / (1))

    for i in range(len(l_w)):
        T_w_occupied[i][step] = (
            T_w_occupied[i][step - 1] + time_step * dT_w_dt_occupied[i]
        )
    for i in range(len(l_r)):
        T_r_occupied[i][step] = (
            T_r_occupied[i][step - 1] + time_step * dT_r_dt_occupied[i]
        )
    for i in range(len(l_f)):
        T_f_occupied[i][step] = (
            T_f_occupied[i][step - 1] + time_step * dT_f_dt_occupied[i]
        )
    T_fur_occupied[step] = T_fur_occupied[step - 1] + time_step * dT_fur_dt_occupied


plt.figure(figsize=(10, 6))
plt.plot(time, rho_b_occupied, label="rho building occupied")

plt.xlabel("Time (hours)")
plt.ylabel("Density (kg/m^3)")
plt.title("Temperature Variations of Internal Elements - Occupied")
plt.legend()
plt.grid(True)
plt.show()
