'''
Generate a map of the excitation probability for chirped laser excitation
as a function of detuning \Delta_0^* and Rabi frequency \Omega^*.

Author: Peter Martin Kruyt
Date: 2025-05-01
'''


import numpy as np
from scipy.integrate import solve_ivp
import json
import matplotlib.pyplot as plt


# Define the function to solve the system of differential equations with given parameters
def chirped_laser_star(tau, y, Omega_star, Delta0_star):
    rho11, rho12_real, rho12_imag, rho22 = y
    rho12 = rho12_real + 1j * rho12_imag
    drho11_dtau = -1j * (0.5 * Omega_star * rho12 - 0.5 * Omega_star * np.conj(rho12))
    drho12_dtau = -1j * (0.5 * Omega_star * rho11 + (tau + Delta0_star) * rho12 - 0.5 * Omega_star * rho22)
    drho22_dtau = -1j * (0.5 * Omega_star * np.conj(rho12) - 0.5 * Omega_star * rho12)
    return [drho11_dtau.real, drho12_dtau.real, drho12_dtau.imag, drho22_dtau.real]

# Initial conditions: population in the ground state at tau = -10
tau0 = -10
y0   = [1, 0, 0, 0]

# Time points where solution is computed
tau_span = (tau0, 10)
tau_eval = np.linspace(tau0, 10, 100)

# Define the function to get the steady-state excitation for given parameters
def get_steady_state_excitation(Delta0_star, Omega_star):
    sol = solve_ivp(chirped_laser_star, tau_span, y0, args=(Omega_star, Delta0_star), t_eval=tau_eval,
                     method='LSODA')
    rho22_sol = sol.y[3]
    steady_state_value = np.mean(rho22_sol[-1])
    return steady_state_value  # Return the steady-state value

# Generate the parameter grid
Delta0_star_range = np.linspace(0, 50, 60)
Omega_star_range  = np.linspace(0, 10, 60)
Delta0_stars, Omega_stars = np.meshgrid(Delta0_star_range, Omega_star_range)

# Calculate the steady-state excitation for each pair of parameters
steady_state_excitation = np.vectorize(get_steady_state_excitation)(Delta0_stars, Omega_stars)

# Plot the heatmap
plt.figure(figsize=(10, 8))
plt.contourf(Delta0_stars, Omega_stars, steady_state_excitation, cmap='viridis', levels=100)
plt.colorbar(label='Excitation probability')
plt.xlabel(r'$\Delta_0^*=\Delta_0/\sqrt{r}$')
plt.ylabel(r'$\Omega^*=\Omega/\sqrt{r}$')
plt.title('Steady-State Excitation Probability')
plt.show()

with open('chirped_excitation_map.json', "w") as f:
    json.dump({
        'Omega_star_max': Omega_star_range[-1],
        'Detuning_star_max': Delta0_star_range[-1],
        'Excitation': steady_state_excitation.tolist()
    }, f, indent=1)

print('Saved.')