# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

# This file will simulate electron cooling in CERN machines
# assuming simple linear optics for the ring, and Parkhomchuk model for the cooler.
#
# WARNING: this is a simple example, not intended to be accurate for real machine simulations.
#          Please refer to literature for more accurate/up-todate parameters.


###### IMPORTS ######
import numpy as np
import xtrack as xt
import xobjects as xo
import xpart as xp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.constants import c as clight

###### Simulation parameters ######
# Time length of tracking e.g. simulate 10 s of cooling, and take data once every 10 ms
max_time_s = 1
int_time_s = 0.01
# number of particles
n_part = 10000
# which machine to simulate
machine = 'AD_300MeV_c'
machine = 'LEIR_Pb'
machine = 'ELENA_100keV'



###### List of Known Machines and their parameters ###### 
machine_parameters = {
    'AD_300MeV_c': { 
         # from https://acc-models.web.cern.ch/acc-models/ad/scenarios/lowenergy/lowenergy.tfs
        'gamma_rel': 1.04987215550, 
        'ring_C':      182.4328,
        'qx':     5.45020077392,
        'qy':     5.41919929346,
        'dqx':  -20.10016919292*0.304548,
        'dqy':  -22.29552755573*0.304548,
        # optics at e-cooler (approximate), in m
        'beta_x': 10,
        'beta_y':  4,
        'alpha_x': 0,
        'alpha_y': 0,
        'D_x':     0,
        # electron cooler parameters:
        'EC_I':          0.1, # A current
        'EC_L':          1.5, # m cooler length
        'EC_beam_r': 25*1e-3, # radius of the electron beam, in m
        'EC_T_perp':  100e-3, # <E> [eV] = kb*T
        'EC_T_long':    1e-3, # <E> [eV] = kb*T
        'EC_B':        0.060, # 600 Gauss
        'EC_B_quality': 1e-4, # rms transverse/longitudinal component.
        'EC_SC_factor':  0.5, # space charge factor: 0: off, 1: fully on
        # some initial beam parameters
        'emittance_x':  4e-6, # geometric, m rad
        'emittance_y':  4e-6, # geometric, m rad
        'dp_p':         3e-4, 
        'q0':             -1, # particle charge in units of e; here pbars
        'mass0_eV_c2': xp.PROTON_MASS_EV, # actual mass at rest of the reference particle
        },
    'AD_100MeV_c': {
        'gamma_rel': 1.00566349, 
        'ring_C':      182.4328,
        'qx':     5.45020077392,
        'qy':     5.41919929346,
        'dqx':  -20.10016919292*0.304548,
        'dqy':  -22.29552755573*0.304548,
        # optics at e-cooler (approximate), in m
        'beta_x': 10,
        'beta_y':  4,
        'alpha_x': 0,
        'alpha_y': 0,
        'D_x':     0,
        # electron cooler parameters:
        'EC_I':          0.1, # A current
        'EC_L':          1.5, # m cooler length
        'EC_beam_r': 25*1e-3, # radius of the electron beam, in m
        'EC_T_perp':  100e-3, # <E> [eV] = kb*T
        'EC_T_long':    1e-3, # <E> [eV] = kb*T
        'EC_B':        0.060, # 600 Gauss
        'EC_B_quality': 1e-4, # rms transverse/longitudinal component.
        'EC_SC_factor':  0.5, # space charge factor: 0: off, 1: fully on
        # some initial beam parameters
        'emittance_x':  1e-6, # geometric, m rad
        'emittance_y':  1e-6, # geometric, m rad
        'dp_p':         2e-4, 
        'q0':             -1, # particle charge in units of e; here pbars
        'mass0_eV_c2': xp.PROTON_MASS_EV, # actual mass at rest of the reference particle
        },
    'ELENA_100keV': {
        'gamma_rel': 1.0001067, 
        'ring_C':   30.40531277976,
        'qx':     2.36168984503,
        'qy':     1.38992572490,
        'dqx':  -25.93420025117*0.1059786,
        'dqy':  -14.90495652098*0.1059786,
        # optics at e-cooler (approximate), in m
        'beta_x':  1.7,
        'beta_y':  2.7,
        'alpha_x': 0,
        'alpha_y': 0,
        'D_x':     1,
        # electron cooler parameters:
        'EC_I':      0.00034, # A current
        'EC_L':           1., # m cooler length
        'EC_beam_r':   14e-3, # radius of the electron beam, in m
        'EC_T_perp':  100e-3, # <E> [eV] = kb*T
        'EC_T_long':    1e-3, # <E> [eV] = kb*T
        'EC_B':        0.010, # 100 Gauss
        'EC_B_quality': 1e-3, # rms transverse/longitudinal component.
        'EC_SC_factor':    1, # space charge factor: 0: off, 1: fully on
        # some initial beam parameters
        'emittance_x':2.5e-6, # geometric, m rad
        'emittance_y':2.5e-6, # geometric, m rad
        'dp_p':         1e-3, 
        'q0':             -1, # particle charge in units of e; here pbars
        'mass0_eV_c2': xp.PROTON_MASS_EV, # actual mass at rest of the reference particle
        },
    'LEIR_Pb':
        { #https://acc-models.web.cern.ch/acc-models/leir/2021/scenarios/nominal/1_flat_bottom/
        'gamma_rel': 1.004510035, 
        'ring_C':   78.54370266,
        'qx':     0.8132925951,
        'qy':     0.7039679756,
        'dqx':  -0.1331854889,
        'dqy':  -0.09978779374,
        # optics at e-cooler (approximate), in m
        'beta_x':  5,
        'beta_y':  5,
        'alpha_x': 0,
        'alpha_y': 0,
        'D_x':     0,
        # electron cooler parameters:
        'EC_I':        0.300, # A current
        'EC_L':          2.5, # m cooler length
        'EC_beam_r': 25*1e-3, # radius of the electron beam, in m
        'EC_T_perp':  100e-3, # <E> [eV] = kb*T
        'EC_T_long':    1e-3, # <E> [eV] = kb*T
        'EC_B':        0.075, # 750 Gauss
        'EC_B_quality': 1e-5, # rms transverse/longitudinal component.
        'EC_SC_factor': 0.75, # space charge factor: 0: off, 1: fully on
        # some initial beam parameters
        'emittance_x': 14e-6, # geometric, m rad
        'emittance_y': 14e-6, # geometric, m rad
        'dp_p':      1.15e-3, # Note: LEIR typical momentum spread should +-2e-3 uniform
        'q0':             54, # particle charge in units of e; here PB54+
        'mass0_eV_c2': xp.Pb208_MASS_EV - 54*xp.ELECTRON_MASS_EV, # actual mass at rest of the reference particle
        },
}




####### FUNCTION DEFINITIONS ######
# Small function to generate matched beam (4D only, with dispersion)
def beam_coordinates(emittance_x, beta_x, alpha_x, 
                     emittance_y, beta_y, alpha_y, 
                     D_x=0, D_y=0, D_px=0, D_py=0, sigma_dp=0,
                     n_particles = 100):
    ''' Returns the x, px, y, py, dp_p coordinates of Gaussian beam matched to 
        the given Twiss parameters (beta, alpha, geometric emittance, in both planes).
        It also include, if non-zero, dispersion contributions to particle coordinates.
    ''' 
    
    # generate betatronic components
    gamma_x = (1+alpha_x**2)/beta_x
    gamma_y = (1+alpha_y**2)/beta_y
    sigma_matrix = np.array(
        [[ beta_x*emittance_x, -alpha_x*emittance_x, 0, 0],
         [-alpha_x*emittance_x, gamma_x*emittance_x, 0, 0],
         [0, 0,  beta_y*emittance_y, -alpha_y*emittance_y],
         [0, 0, -alpha_y*emittance_y, gamma_y*emittance_y]
         ])
    betatronic_motion = np.random.multivariate_normal((0, 0, 0, 0), sigma_matrix, n_particles).transpose()
    
    # generate momentum deviations
    dp_p = np.random.normal(0, sigma_dp, n_particles)

    ## generate dispersive components of motion
    dispersive_motion = np.outer(np.array([D_x, D_px, D_y, D_py]),dp_p)

    ## return total coordinates
    return np.vstack(((betatronic_motion + dispersive_motion), dp_p))
####### END OF FUNCTION DEFINITION ######


# extract parameters for choosen machine
gamma_rel   = machine_parameters[machine]['gamma_rel']
ring_C      = machine_parameters[machine]['ring_C']
qx          = machine_parameters[machine]['qx']
qy          = machine_parameters[machine]['qy']
dqx         = machine_parameters[machine]['dqx']
dqy         = machine_parameters[machine]['dqy']
beta_x      = machine_parameters[machine]['beta_x']
beta_y      = machine_parameters[machine]['beta_y']
alpha_x     = machine_parameters[machine]['alpha_x']
alpha_y     = machine_parameters[machine]['alpha_y']
D_x         = machine_parameters[machine]['D_x']
EC_current  = machine_parameters[machine]['EC_I']
EC_length   = machine_parameters[machine]['EC_L']
EC_beam_r   = machine_parameters[machine]['EC_beam_r']
temp_perp   = machine_parameters[machine]['EC_T_perp']
temp_long   = machine_parameters[machine]['EC_T_long']
EC_B        = machine_parameters[machine]['EC_B']
EC_B_ratio  = machine_parameters[machine]['EC_B_quality']
EC_SC_factor= machine_parameters[machine]['EC_SC_factor']
emittance_x = machine_parameters[machine]['emittance_x']
emittance_y = machine_parameters[machine]['emittance_y']
dp_p        = machine_parameters[machine]['dp_p']
q0          = machine_parameters[machine]['q0']
mass0       = machine_parameters[machine]['mass0_eV_c2']

# compute derived beam parameters
beta_rel    = np.sqrt(gamma_rel**2 - 1)/gamma_rel
p0c         = mass0*beta_rel*gamma_rel #eV/c
T_per_turn  = ring_C/(clight*beta_rel)
beta_gamma  = beta_rel*gamma_rel
#
nemitt_x = emittance_x*beta_gamma
nemitt_y = emittance_x*beta_gamma

# compute length of simulation, as well as sample interval, in turns
num_turns     = int(max_time_s/T_per_turn)
save_interval = int(int_time_s/T_per_turn)

##### DEFINE LATTICE #####
# Define the whole machine a single arc with given linear optics
arc = xt.LineSegmentMap(
        qx=qx, qy=qx,
        dqx=dqx, dqy=dqy,
        length=ring_C,
        betx=beta_x,
        bety=beta_y,
        alfx=alpha_x,
        alfy=alpha_y,
        dx=D_x)
# Define the electron cooler
electron_cooler = xt.ElectronCooler(
        length        = EC_length,
        current       = EC_current,
        temp_perp     = temp_perp,
        temp_long     = temp_long,
        radius_e_beam = EC_beam_r,
        offset_x      = 0.0,
        offset_px     = 0.0,
        offset_y      = 0.0,
        offset_py     = 0.0,
        offset_energy = 0.0,
        magnetic_field      = EC_B, 
        magnetic_field_ratio= EC_B_ratio,
        space_charge_factor = EC_SC_factor)
# create a monitor object, to reduce holded data
monitor = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=1,
                        n_repetitions=int(num_turns/save_interval),
                        repetition_period=save_interval,
                        num_particles=n_part)

# assemble line and build its tracker
line = xt.Line(
        elements=[monitor, electron_cooler, arc])
# add reference particle
line.particle_ref = xp.Particles(mass0=mass0, q0=q0, p0c=p0c)
# create a context
context = xo.ContextCpu(omp_num_threads='auto')
# build a tracker
line.build_tracker(_context = context)


##### CREATE BEAM #####
coordinates = beam_coordinates(
        emittance_x, beta_x, alpha_x,
        emittance_y, beta_y, alpha_y,
        D_x=D_x, D_y=0, D_px=0, D_py=0,
        sigma_dp=dp_p,
        n_particles = n_part)
particles = line.build_particles(
        num_particles=n_part, weight=1,
        x     = coordinates[0,:],
        px    = coordinates[1,:],
        y     = coordinates[2,:],
        py    = coordinates[3,:],
        delta = coordinates[4,:], 
        zeta  = 0)


##### ACTUAL TRACKING #####
#  track all particles, *without* keeping turn-by-turn data (too memory expensive!)
line.track(particles, num_turns=num_turns,
        turn_by_turn_monitor=False, with_progress=True)


##### DATA ANALYSIS AND PLOT #####
# extract relevant values from monitor object
#  [all times, all particles, first (and only) monitor]
out_x     = monitor.x[:,:,0]
out_px    = monitor.px[:,:,0]
out_y     = monitor.y[:,:,0]
out_py    = monitor.py[:,:,0]
out_delta = monitor.delta[:,:,0]
out_time  = monitor.at_turn[:, 0, 0] * T_per_turn

# compute single particle actions:
#   - for x, remove the dp/p contribution before computing
#   - for y, simple compututation:
gamma_x = (1+alpha_x**2)/beta_x
out_action_x = (gamma_x*(out_x-D_x*out_delta)**2 + 2*alpha_x*(out_x-D_x*out_delta)*(out_px) + beta_x*(out_px)**2)/2
gamma_y = (1+alpha_y**2)/beta_y
out_action_y = (gamma_y*(out_y)**2 + 2*alpha_y*(out_y)*(out_py) + beta_y*out_py**2)/2
# compute emittances as average actions
out_emittance_x=np.mean(out_action_x, axis=1)
out_emittance_y=np.mean(out_action_y, axis=1)
# compute beam sizes and dp/p
out_sigma_x=np.std(out_x*1e3, axis=1)
out_sigma_y=np.std(out_y*1e3, axis=1)
out_sigma_p=np.std(out_delta, axis=1)

# different approach: compute 95% emittance (factor 2 to be checked...)
out_emittance_x_95 = np.quantile(out_action_x, 0.95, axis=1)
out_emittance_y_95 = np.quantile(out_action_y, 0.95, axis=1)
out_sigma_p_95     = np.quantile(np.abs(out_delta), 0.95, axis=1)

##### produce some plots #####
# plot rms emittances and dp/p
plt.figure(figsize=(10,8))
plt.subplot(2,1,1)
plt.plot(out_time, out_emittance_x*1e6, label=f'$\epsilon_x$ (init={emittance_x*1e6})')
plt.plot(out_time, out_emittance_y*1e6, label=f'$\epsilon_y$ (init={emittance_y*1e6})')
plt.ylim([0, np.max((emittance_x, emittance_y))*1e6*1.05])
plt.ylabel('RMS $\epsilon$ [$\mu$m]')
plt.xlabel('Time [s]')
plt.legend()

plt.subplot(2,1,2)
plt.plot(out_time, out_sigma_p*1e4, label=f'RMS dp/p (init={dp_p*1e4})')
plt.ylim([0, dp_p*1e4*1.05])
plt.ylabel('RMS dp/p [$10^{-4}$]')
plt.xlabel('Time [s]')
plt.legend()

plt.show()


# plot emittances and dp/p using 95% definition
plt.figure(figsize=(10,8))

plt.subplot(2,1,1)
plt.plot(out_time, out_emittance_x_95*1e6, label=f'$\epsilon_x$')
plt.plot(out_time, out_emittance_y_95*1e6, label=f'$\epsilon_y$')
plt.ylim([0, 3*np.max((emittance_x, emittance_x))*1e6*1.05])
plt.ylabel('95% $\epsilon$ [$\mu$m]')
plt.xlabel('Time [s]')
plt.legend()

plt.subplot(2,1,2)
plt.plot(out_time, out_sigma_p_95*1e4, label=f'dp/p')
plt.ylim([0, 2*dp_p*1e4*1.05])
plt.ylabel('95% dp/p [$10^{-4}$]')
plt.xlabel('Time [s]')
plt.legend()

plt.show()


# 2D histogram
plt.figure(figsize=(10, 10))
plt.subplot(3,1,1)
h=plt.hist2d(np.repeat(out_time, n_part), out_x.flatten(), 
           bins=[out_time, np.max(np.abs(out_x))*np.linspace(-1, 1, 1+int(n_part/100))], 
           cmap='turbo', norm=LogNorm(vmin=0.1), density=False)
plt.xlabel('Time [s]')
plt.ylabel('x [m]')
plt.colorbar(label='# Particles')
plt.tight_layout()

plt.subplot(3,1,2)
h=plt.hist2d(np.repeat(out_time, n_part), out_y.flatten(), 
           bins=[out_time, np.max(np.abs(out_y))*np.linspace(-1, 1, 1+int(n_part/100))], 
           cmap='turbo', norm=LogNorm(vmin=0.1), density=False)
plt.xlabel('Time [s]')
plt.ylabel('y [m]')
plt.colorbar(label='# Particles')
plt.tight_layout()

plt.subplot(3,1,3)
h=plt.hist2d(np.repeat(out_time, n_part), out_delta.flatten(), 
           bins=[out_time, np.max(np.abs(out_delta))*np.linspace(-1, 1, 1+int(n_part/100))], 
           cmap='turbo', norm=LogNorm(vmin=0.1), density=False)
plt.xlabel('Time [s]')
plt.ylabel('dp/p')
plt.colorbar(label='# Particles')
plt.tight_layout()

plt.show()

###### END OF SCRIPT ######