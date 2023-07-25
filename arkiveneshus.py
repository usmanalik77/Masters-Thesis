import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint
import pandas as pd
import csv
import datetime

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
R = 8.314  # Pa*m^3 /(mol*K)
M_b = 0.02897  # kg/mol
T_b_occupied_initial = 293.15
T_b_unoccupied_initial = 288.15
# T_b_occupied_initial = 0.05
# T_b_unoccupied_initial = 0.05

time_i = np.arange(0, duration, time_step)
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
# print(rho_b_occupied)
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
dT_b_dt_occupied = np.zeros(num_steps)
dQ_dt_supply = np.zeros(num_steps)
T_infinity = T_outside
Insolation = 100  # W/m^2
Appliances = 150  # W
dQ_people_dt = 58.2 * area_room  # W
dQ_people_dt_series = np.zeros(num_steps)
dQ_appliances_dt = Appliances  # W
dQ_solar_dt = Insolation * area_room
Q_heater = 1000  # W]
dQ_heater_dt = np.zeros(num_steps)
# dQ_heater_dt = Q_heater
k1_occupied = np.zeros(num_steps)
k2_occupied = np.zeros(num_steps)
dQ_supply_dt = dQ_appliances_dt + dQ_solar_dt + dQ_people_dt + Q_heater
T_b_occupied_unbiased = np.zeros(num_steps)
T_b_occupied_unbiased[0] = T_b_occupied_initial

# Data from the arkivenes hus building
A_window = 1996 / 1000  # NOTE: Combined with Adoor! 1.2 # [m^2] Area of window
A_door = 1996 / 1000  # 0.75*2.1 # [m^2] Area of door
A_walls = (
    3486 / 1000
)  # 2*(self.l+self.w)*self.h - (self.Awindow+self.Adoor) # [m^2] Area of walls
A_wallsBase = 1878 / 1000  # [m^2] Area of walls
A_roof = 3090 / 1000  # self.l*self.w # [m^2] Area of roof
A_floor = 2269 / 1000  #  self.l*self.w # [m^2] Area of floor

# Ventilation
V_in = (1 * 1375) / (A_floor)  # (airflow )/(3600*(self.Afloor)) #(self.l*self.w)
V_out = V_in
# Overall heat transfer coefficients
U_window = 0  # NOTE: Combined with Adoor!  1.2 # [W/m^2K] Heat transfer coeff of window
U_door = 0.8  # 1.2 # [W/m^2K] Heat transfer coeff of door
U_walls = 0.17  # 0.18 # [W/m^2K] Heat transfer coeff of walls
U_wallsBase = 0.15  # [W/m^2K] Heat transfer coeff of walls
U_roof = 0.13  # [W/m^2K] Heat transfer coeff of roof
U_floor = 0.18  #  0.15 # [W/m^2K] Heat transfer coeff of floor

Q_solar = np.zeros(num_steps)
dQ_supply_dt = np.zeros(num_steps)
for i in range(num_steps):
    if i >= 10 * 1 / 60 * 100 and i <= 14 * 1 / 60 * 100:
        Q_solar[i] = 100 * A_window * 1
    else:
        Q_solar[i] = 0
    dQ_supply_dt[i] = dQ_appliances_dt + Q_solar[i] + dQ_people_dt + Q_heater

for i in range(1, num_steps):
    dQ_window_dt_occupied = U_window * A_window * (T_infinity)
    dQ_door_dt_occupied = U_door * A_door * (T_infinity)
    dQ_walls_dt_occupied = U_walls * A_walls * (T_infinity)
    dQ_floor_dt_occupied = U_floor * A_floor * (T_infinity)
    dQ_roof_dt_occupied = U_roof * A_roof * (T_infinity)

    dQ_dt_loss_occupied = (
        dQ_window_dt_occupied
        + dQ_door_dt_occupied
        + dQ_walls_dt_occupied
        + dQ_floor_dt_occupied
        + dQ_roof_dt_occupied
    )

    rho_in_occupied = (P_in * M_b) / (R * T_b_occupied_initial)
    rho_out_occupied = rho_b_occupied
    drho_b_occupied_dt = 0

    if i <= 50:
        # dQ_dt_supply[i] = dQ_supply_dt
        V_in = V_out = 0.7
        # rho_in_occupied = rho_out_occupied = 1.1041
        # rho_in_unoccupied = rho_in_unoccupied = 1.1041
        h_in = 60179.47996
        h_out = 60587.96
        k1_occupied_0 = 70
        k1_occupied[i] = (
            (V_in * rho_in_occupied * h_in - V_out * rho_out_occupied * h_out)
            + dQ_dt_supply[i]
            + dQ_dt_loss_occupied
        ) / (V_b * rho_b_occupied * (c_p_b - R / M_b))
        # k1_occupied[i] = 0.1
        k1 = 70
        k1_unbiased = 70
        k2_occupied[i] = dQ_dt_loss_occupied / T_infinity + (
            V_in * rho_in_occupied - V_out * rho_out_occupied
        ) / (rho_b_occupied * V_b)
        k2 = 0.23
        T_b_occupied[i] = (1 / k2) * (
            k1_occupied_0 - (k1 - k2 * T_b_occupied[0]) * math.exp(-k2 * i)
        )
        T_b_occupied_unbiased[i] = (1 / k2) * (
            k1_occupied_0 - (k1_unbiased - k2 * T_b_occupied[0]) * math.exp(-k2 * i)
        )
        # T_b_occupied[i] =  (k1_occupied_0 - (k1_occupied[i] - k2_occupied[i]*T_b_occupied_initial)*math.exp(-k2_occupied[i]*i))
    else:
        V_in = V_out = 0.7
        # rho_in_occupied = rho_out_occupied = 1.1041
        # rho_in_unoccupied = rho_in_unoccupied = 1.1041
        h_in = 61173.08
        h_out = 60587.96
        # dQ_dt_supply[i] = 0
        k1_occupied_50 = 50
        k1_occupied[i] = (
            (V_in * rho_in_occupied * h_in - V_out * rho_out_occupied * h_out)
            + dQ_dt_supply[i]
            + dQ_dt_loss_occupied
        ) / (V_b * rho_b_occupied * (c_p_b - R / M_b))
        # k1_occupied[i] = 4
        k2 = 0.19
        k1 = 50
        k1_unbiased = 50
        k2_occupied[i] = dQ_dt_loss_occupied / T_infinity + (
            V_in * rho_in_occupied - V_out * rho_out_occupied
        ) / (rho_b_occupied * V_b)
        T_b_occupied[i] = (1 / k2) * (
            k1_occupied_50 - (k1 - k2 * T_b_occupied[50]) * math.exp(-k2 * (i - 50))
        )
        T_b_occupied_unbiased[i] = (1 / k2) * (
            k1_occupied_50
            - (k1_unbiased - k2 * T_b_occupied[50]) * math.exp(-k2 * (i - 50))
        )
        # T_b_occupied[i] =   (k1_occupied_50 -  (k1_occupied[i]-k2_occupied[i]*T_b_occupied[50])*math.exp(-k2_occupied[i]*(i-50)))

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
            ) + (dQ_dt_supply[step] - dQ_dt_loss_occupied) / (rho_f[i] * c_p_f[i])
        elif i == len(l_f) - 1:
            dT_f_dt_occupied[i] = alpha_f[i] * (
                (
                    T_f_occupied[i - 1][step - 1]
                    - 2 * T_f_occupied[i][step - 1]
                    + T_f_occupied[i][step - 1]
                )
                / (l_f[i] ** 2)
            ) + (dQ_dt_supply[step] - dQ_dt_loss_occupied) / (rho_f[i] * c_p_f[i])
        else:
            dT_f_dt_occupied[i] = alpha_f[i] * (
                (
                    T_f_occupied[i + 1][step - 1]
                    - 2 * T_f_occupied[i][step - 1]
                    + T_f_occupied[i][step - 1]
                )
                / (l_f[i] ** 2)
            ) + (dQ_dt_supply[step] - dQ_dt_loss_occupied) / (rho_f[i] * c_p_f[i])

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

# def dydt(y0, time, T_outside):
#     for i in range(1, num_steps):
#         deltaRhoB = (V_in*rho_in-V_out*rho_b_occupied) / V_b

#         firstPart[i] = (V_in*rho_in*h_in - V_out*rho_b_occupied*h_out + dQ_dt_supply[i]-dQ_dt_loss_occupied) / (V_b*rho_b_unoccupied*(c_p_b-(R/M_b))) - (y0/rho_b_occupied)*deltaRhoB
#     return firstPart[i]
K = 273.15
# y0 = 15+K
# tolerance = 1e-5

# sol = odeint(dydt, y0, time, args=(T_outside, ),rtol=tolerance,atol=tolerance)
with open(
    "E:/Python_simulation/i9 data/29_04_21/i9_320001_RT901_Temp.csv", "r"
) as csv_file:
    csv_file_reader1 = csv.reader(csv_file)
    rows1 = list()
    for row in csv_file_reader1:
        rows1.append(row)
# with open(
#     "E:/Python_simulation/i9 data/29_04_21/i9_360005_RT401_Temp.csv", "r"
# ) as csv_file:
#     csv_file_reader2 = csv.reader(csv_file)
#     rows2 = list()
#     for row in csv_file_reader2:
#         rows2.append(row)
with open(
    "E:/Python_simulation/i9 data/29_04_21/i9_360005_RT501_Temp.csv", "r"
) as csv_file:
    csv_file_reader3 = csv.reader(csv_file)
    rows3 = list()
    for row in csv_file_reader3:
        rows3.append(row)
with open(
    "E:/Python_simulation/i9 data/29_04_21/i9_360005_RT901_Temp.csv", "r"
) as csv_file:
    csv_file_reader4 = csv.reader(csv_file)
    rows4 = list()
    for row in csv_file_reader4:
        rows4.append(row)
with open(
    "E:/Python_simulation/i9 data/29_04_21/i9_360006_RT401_Temp.csv", "r"
) as csv_file:
    csv_file_reader5 = csv.reader(csv_file)
    rows5 = list()
    for row in csv_file_reader5:
        rows5.append(row)
with open(
    "E:/Python_simulation/i9 data/29_04_21/i9_360006_RT501_Temp.csv", "r"
) as csv_file:
    csv_file_reader6 = csv.reader(csv_file)
    rows6 = list()
    for row in csv_file_reader6:
        rows6.append(row)

with open(
    "E:/Python_simulation/i9 data/29_04_21/raw_weather_temps.csv", "r"
) as csv_file:
    csv_file_reader7 = csv.reader(csv_file)
    rows7 = list()
    for row in csv_file_reader7:
        rows7.append(row)
# with open(
#     "E:/Python_simulation/i9 data/29_04_21/i9_320001_OE501_Power.csv", "r"
# ) as csv_file:
#     csv_file_reader8 = csv.reader(csv_file)
#     rows8 = list()
#     for row in csv_file_reader8:
#         rows8.append(row)
with open(
    "E:/Python_simulation/i9 data/29_04_21/i9_432001_XZ001_Power.csv", "r"
) as csv_file:
    csv_file_reader9 = csv.reader(csv_file)
    rows9 = list()
    for row in csv_file_reader9:
        rows9.append(row)

with open(
    "E:/Python_simulation/i9 data/29_04_21/i9_360005_RB600_Occupancy.csv", "r"
) as csv_file:
    csv_file_reader10 = csv.reader(csv_file)
    rows10 = list()
    for row in csv_file_reader10:
        rows10.append(row)
with open(
    "E:/Python_simulation/i9 data/29_04_21/i9_360006_RB600_Occupancy.csv", "r"
) as csv_file:
    csv_file_reader11 = csv.reader(csv_file)
    rows11 = list()
    for row in csv_file_reader11:
        rows11.append(row)

with open(
    "E:/Python_simulation/i9 data/29_04_21/i9_360005_RF401_Flow.csv", "r"
) as csv_file:
    csv_file_reader12 = csv.reader(csv_file)
    rows12 = list()
    for row in csv_file_reader12:
        rows12.append(row)
with open(
    "E:/Python_simulation/i9 data/29_04_21/i9_360005_RF501_Flow.csv", "r"
) as csv_file:
    csv_file_reader13 = csv.reader(csv_file)
    rows13 = list()
    for row in csv_file_reader13:
        rows13.append(row)
    # num_lines = sum(1 for row in csv_file_reader if len(row)>0)
    # num_lines_2 = sum(1 for row in csv_file_reader if len(row)>=0)
    # print(num_lines, num_lines_2)
    # time_csv = np.zeros(num_lines)
    # temp_csv = np.zeros(num_lines)
    # csv_file_reader_2 = list()
    # for i, row in zip(range(num_lines_2), csv_file_reader):
    #     csv_file_reader_2[i] = csv_file_reader[i]
    # # print(csv_file_reader_2)
    # csv_file_reader_2 = [row for row in csv_file_reader_2 if len(row) > 0]
    # for i, row in enumerate(csv_file_reader_2):
    #     print(row)
    #     time_csv[i] = int(row[0])
    #     temp_csv[i] = int(row[1])


def time_temp(rows):
    time_csv = list()
    temp_csv = list()
    for i, row in enumerate(rows):
        # print(i, row)
        if i == 0 or len(row) == 0:
            # print(row)
            continue
        time_csv.append(
            datetime.datetime.fromisoformat(row[0]).strftime("%Y%m%d%H%M%S")
        )
        temp_csv.append(float(row[1]))
    time_csv = [int(time) - 20210224140252 for time in time_csv]
    temp_csv = [temp + K for temp in temp_csv]
    return time_csv, temp_csv


def time_power(rows):
    time_csv = list()
    power_csv = list()
    for i, row in enumerate(rows):
        # print(i, row)
        if i == 0 or len(row) == 0:
            # print(row)
            continue
        time_csv.append(
            datetime.datetime.fromisoformat(row[0]).strftime("%Y%m%d%H%M%S")
        )
        power_csv.append(float(row[1]))
    # print(time_csv[0])
    time_csv = [int(time) - 20201026123209 for time in time_csv]
    power_csv = [power for power in power_csv]
    return time_csv, power_csv


def time_people(rows):
    time_csv = list()
    people_csv = list()
    for i, row in enumerate(rows):
        # print(i, row)
        if i == 0 or len(row) == 0:
            # print(row)
            continue
        time_csv.append(
            datetime.datetime.fromisoformat(row[0]).strftime("%Y%m%d%H%M%S")
        )
        people_csv.append(float(row[1]))
    print(time_csv[0])
    time_csv = [int(time) - 20210224134751 for time in time_csv]
    people_csv = [people for people in people_csv]
    return time_csv, people_csv


def time_flow(rows):
    time_csv = list()
    flow_csv = list()
    for i, row in enumerate(rows):
        # print(i, row)
        if i == 0 or len(row) == 0:
            # print(row)
            continue
        time_csv.append(
            datetime.datetime.fromisoformat(row[0]).strftime("%Y%m%d%H%M%S")
        )
        flow_csv.append(float(row[1]))
    print(time_csv[0])
    time_csv = [int(time) - 20210224134751 for time in time_csv]
    flow_csv = [people for people in flow_csv]
    return time_csv, flow_csv


# print(time_csv[0], temp_csv[0])
# read_csv = pd.read_csv("D:/Python_simulation/i9 data/29_04_21/i9_360006_RT501_Temp.csv")
# num_lines = sum(1 for row in read_csv if len(row)>0)
# num_lines_2 = sum(1 for row in read_csv if len(row)>=0)
# print(num_lines, num_lines_2)
# time_csv = np.zeros(num_lines)
# temp_csv = np.zeros(num_lines)
# # csv_file_reader_2 = list()
# # for i, row in zip(range(num_lines_2), read_csv):
# #     csv_file_reader_2[i] = read_csv[i]
# # print(csv_file_reader_2)
# # csv_file_reader_2 = [row for row in csv_file_reader_2 if len(row) > 0]
# for i, row in enumerate(read_csv):
#     print(row)
#     # time_csv[i] = int(row)
#     # temp_csv[i] = int(row)

# print(time_csv, temp_csv)s
time_csv1, temp_csv1 = time_temp(rows1)
time_csv1, temp_csv1 = zip(*sorted(zip(time_csv1, temp_csv1)))
# time_csv2, temp_csv2 = time_temp(rows2)
# time_csv2, temp_csv2 = zip(*sorted(zip(time_csv2, temp_csv2)))
time_csv3, temp_csv3 = time_temp(rows3)
time_csv3, temp_csv3 = zip(*sorted(zip(time_csv3, temp_csv3)))
time_csv4, temp_csv4 = time_temp(rows4)
time_csv4, temp_csv4 = zip(*sorted(zip(time_csv4, temp_csv4)))
time_csv5, temp_csv5 = time_temp(rows5)
time_csv5, temp_csv5 = zip(*sorted(zip(time_csv5, temp_csv5)))
time_csv6, temp_csv6 = time_temp(rows6)
time_csv6, temp_csv6 = zip(*sorted(zip(time_csv6, temp_csv6)))
time_csv7, temp_csv7 = time_temp(rows7)
time_csv7, temp_csv7 = zip(*sorted(zip(time_csv7, temp_csv7)))
# time_csv8, power_csv8 = time_power(rows8)
# time_csv8, power_csv8 = zip(*sorted(zip(time_csv8, power_csv8)))
time_csv9, power_csv9 = time_power(rows9)
time_csv9, power_csv9 = zip(*sorted(zip(time_csv9, power_csv9)))
time_csv10, people_csv10 = time_people(rows10)
time_csv10, people_csv10 = zip(*sorted(zip(time_csv10, people_csv10)))
time_csv11, people_csv11 = time_people(rows11)
time_csv11, people_csv11 = zip(*sorted(zip(time_csv11, people_csv11)))
time_csv12, flow_csv12 = time_flow(rows12)
time_csv12, flow_csv12 = zip(*sorted(zip(time_csv12, flow_csv12)))
time_csv13, flow_csv13 = time_flow(rows13)
time_csv13, flow_csv13 = zip(*sorted(zip(time_csv13, flow_csv13)))

time_csv81 = list()
power_csv81 = list()
# for i, (time, power) in enumerate(zip(time_csv8, power_csv8)):
#     if time <= time_csv2[-1]:
#         time_csv81.append(time)
#         power_csv81.append(power)
#     else:
#         continue

time_csv101 = list()
people_csv101 = list()
# for i, (time, people) in enumerate(zip(time_csv10, people_csv10)):
#     if time <= time_csv2[-1]:
#         time_csv101.append(time)
#         people_csv101.append(people)
#     else:
#         continue

# time_csv121 = list()
# flow_csv121 = list()
# for i, (time, flow) in enumerate(zip(time_csv8, flow_csv12)):
#     if time <= time_csv2[-1]:
#         time_csv121.append(time)
#         flow_csv121.append(flow)
#     else:
#         continue


# print(time_csv10)

plt.figure(figsize=(10, 6))
# fig, ax1 = plt.subplots()

# color = 'tab:red'
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('"T Arkivenes Hus 320001_RT901 [deg K]"', color=color)
# ax1.plot(time_csv2, temp_csv2, color=color)
# ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# color = 'tab:blue'
# ax2.set_ylabel('Power Arkivenes Hus 320001_OE501 [W]', color=color)  # we already handled the x-label with ax1
# ax2.plot(time_csv81, power_csv81, color=color)
# ax2.tick_params(axis='y', labelcolor=color)
# plt.grid(True)
# #######################################################
# fig1, ax2 = plt.subplots()

# color = 'tab:red'
# ax2.set_xlabel('Time (s)')
# ax2.set_ylabel('"T Arkivenes Hus 320001_RT901[deg K]"', color=color)
# ax2.plot(time_csv2, temp_csv2, color=color)
# ax2.tick_params(axis='y', labelcolor=color)

# ax3 = ax2.twinx()  # instantiate a second axes that shares the same x-axis

# color = 'tab:blue'
# ax3.set_ylabel('People Arkivenes Hus 360005_RB600', color=color)  # we already handled the x-label with ax1
# ax3.plot(time_csv101, people_csv101, color=color)
# ax3.tick_params(axis='y', labelcolor=color)
# #######################################################
# fig2, ax4 = plt.subplots()

# color = 'tab:red'
# ax4.set_xlabel('Time (s)')
# ax4.set_ylabel('"T Arkivenes Hus 320001_RT901 [deg K]"', color=color)
# ax4.plot(time_csv2, temp_csv2, color=color)
# ax4.tick_params(axis='y', labelcolor=color)

# ax5 = ax4.twinx()  # instantiate a second axes that shares the same x-axis

# color = 'tab:blue'
# ax5.set_ylabel('Flow Arkivenes Hus 360005_RF401 [kg/s]', color=color)  # we already handled the x-label with ax1
# ax5.plot(time_csv121, flow_csv121, color=color)
# ax5.tick_params(axis='y', labelcolor=color)
# plt.plot(time_csv1, temp_csv1, label="T Arkivenes Hus 320001_RT901")
# plt.plot(time_csv2, temp_csv2, label="T Arkivenes Hus 360005_RT401")
# plt.plot(time_csv3, temp_csv3, label="T Arkivenes Hus 360005_RT501")
# plt.plot(time_csv4, temp_csv4, label="T Arkivenes Hus 360005_RT901")
# plt.plot(time_csv5, temp_csv5, label="T Arkivenes Hus 360006_RT401")
# plt.plot(time_csv6, temp_csv6, label="T Arkivenes Hus 360006_RT501")
# plt.plot(time_csv1, temp_csv1, label="T Arkivenes Hus 320001_RT901")
# plt.plot(time_csv2, temp_csv2, label="T Arkivenes Hus 360005_RT401")
# plt.plot(time_csv3, temp_csv3, label="T Arkivenes Hus 360005_RT501")
# plt.plot(time_csv4, temp_csv4, label="T Arkivenes Hus 360005_RT901")
# plt.plot(time_csv5, temp_csv5, label="T Arkivenes Hus 360006_RT401")
# plt.plot(time_csv6, temp_csv6, label="T Arkivenes Hus 360006_RT501")

# plt.plot(time_csv7, temp_csv7, label="Raw weather Temperature")
# plt.plot(time_csv8, power_csv8, label="Power Arkivenes Hus 320001_OE501")
# plt.plot(time_csv9, power_csv9, label="Power Arkivenes Hus 432001_XZ001")

# plt.plot(time_csv10, people_csv10, label="People Arkivenes Hus 360005_RB600")
# plt.plot(time_csv11, people_csv11, label="People Arkivenes Hus 360006_RB600")
plt.plot(time_i, T_b_occupied, label="T building occupied")
# plt.plot(time, sol, label="T numerical building occupied")
# plt.plot(time, T_b_occupied_unbiased, label="T different coefficients building occupied")
for i in range(len(l_w)):
    plt.plot(time_i, T_w_occupied[i], label=f"Wall Layer {i+1} occupied")
for i in range(len(l_r)):
    plt.plot(time_i, T_r_occupied[i], label=f"Roof Layer {i+1} occupied")
for i in range(len(l_f)):
    plt.plot(time_i, T_f_occupied[i], label=f"Floor Layer {i+1} occupied")
plt.plot(time_i, T_fur_occupied, label="Furniture occupied")
plt.xlabel("Time (hours)")
plt.ylabel("Temperature (°K)")
# plt.ylabel('Power (W)')
# plt.ylabel('People')
plt.title("Temperature Variations of Internal Elements - Occupied")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
#             ncol=1, mode="expand", borderaxespad=0.)

plt.legend()
plt.grid(True)
plt.show()
