# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

# This file will simulate laser cooling in SPS as part of the PoP experiment.


###### IMPORTS ######
import numpy as np
import xtrack as xt
import xobjects as xo
import xpart as xp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
from scipy.constants import c as clight
from scipy import constants

# Import ion properties
from ion_properties import lead as ion

###### Simulation parameters ######
# Time length of tracking e.g. simulate 10 s of cooling, and take data once every 10 ms
max_time_s = 100
int_time_s = 0.01
# number of particles
n_part = 1000


###### Define the machine ######
#### FROM HARD-CODED PARAMETERS ######

# machine parameters for SPS
ring_C = 6911.5038  # m
qx   = 26.299364685600626
qy   = 26.24936295006925
dqx  = -0.13570110411365022
dqy  = -0.09261358908219108
qs   = 0.0062838580355705775
bets = 313.73880787603525
momentum_compaction_factor = 0.00190098472346712

#at interaction points: #from https://anaconda.org/petrenko/li_like_ca_in_sps/notebook
beta_x  =  54.614389 # m
beta_y  =  44.332517 # m
alpha_x = -1.535235
alpha_y =  1.314101
Dx      =  2.444732 # m
Dpx     =  0.097522
Dy      =  0.0 # m
Dpy     =  0.0

if False:
        #### FROM JSON FILE of SPS - if you have it available ######
        with open('sps.json', 'r') as fid:
                line = xt.Line.from_json(fid)
        particle_ref=line.particle_ref
        line.build_tracker()
        twiss=line.twiss()

        ring_C = line.get_length()
        qx   = twiss['qx']
        qy   = twiss['qy']
        dqx  = twiss['dqx']
        dqy  = twiss['dqy']
        qs   = twiss['qs']
        bets = twiss['bets0']
        momentum_compaction_factor = twiss['momentum_compaction_factor']

        #Note: index of gamma factory along SPS line should be around idx: 16675
        _idx_PoP = 16675
        beta_x  =  twiss['betx'][_idx_PoP]
        beta_y  =  twiss['bety'][_idx_PoP]
        alpha_x =  twiss['alfx'][_idx_PoP]
        alpha_y =  twiss['alfy'][_idx_PoP]
        Dx      =    twiss['dx'][_idx_PoP]
        Dpx     =   twiss['dpx'][_idx_PoP]
        Dy      =    twiss['dy'][_idx_PoP]
        Dpy     =   twiss['dpy'][_idx_PoP]
##########################################


# Other, RF parameters (Probably not used)
voltage_rf = 7*1e6
frequency  = 201.8251348335775*1e6
lag_rf     = 180


# Extract ion parameters
q0        = ion.q0
mass0     = ion.mass0
gamma_rel = ion.gamma_cooling
bunch_intensity = ion.N_a
nemitt    = 1.5e-6 # m*rad (normalized emittance)
sigma_z   = 0.063 # m

# compute derived beam parameters
beta_rel  = np.sqrt(1-1/(gamma_rel*gamma_rel))
beta_gamma= beta_rel*gamma_rel
p0c       = mass0*beta_gamma #eV/c
gemitt    = nemitt/beta_gamma

#############
# Generate reference particle 
particle_ref = xp.Particles(p0c=p0c, mass0=mass0, q0=q0, gamma0=gamma_rel)
#############

# compute length of simulation, as well as sample interval, in turns
T_per_turn    = ring_C/(clight*beta_rel)
num_turns     = int(max_time_s/T_per_turn)
save_interval = int(int_time_s/T_per_turn)


##### DEFINE LATTICE #####

##################
# ARC            #
##################
# Define the whole machine a single arc with given linear optics
arc = xt.LineSegmentMap(
        qx    = qx, 
        qy    = qy,
        dqx   = dqx, 
        dqy   = dqy,
        length= ring_C,
        alfx  = alpha_x,
        alfy  = alpha_y,
        betx  = beta_x,
        bety  = beta_y,
        dx    = Dx,
        dpx   = Dpx,
        dy    = Dy,
        dpy   = Dpy,
        qs    = qs,
        bets  = bets,
        # momentum_compaction_factor = momentum_compaction_factor,
        # voltage_rf=voltage_rf,
        # lag_rf=lag_rf,
        # frequency_rf=frequency,
        # longitudinal_mode = 'nonlinear',
        )


############################################################
# Assmple arc line for matched bunch generation
# Need to it now, so I get to have info about beam parameters
# needed to optimise laser cooling....
# TODO: refactor to avoid code duplication
line_arc=xt.Line(
        elements=[arc])
line_arc.particle_ref=particle_ref

line_arc.build_tracker()
# Generate matched bunch
particles = xp.generate_matched_gaussian_bunch(
        num_particles=n_part,
        total_intensity_particles=bunch_intensity,
        nemitt_x=nemitt, nemitt_y=nemitt, sigma_z=sigma_z,
        particle_ref=particle_ref,
        line=line_arc,
        #ngine="single-rf-harmonic",        
        )
particles._init_random_number_generator()
particles0=particles.copy()

# compute initial dp/p
sigma_dp=np.std(particles.delta)
print('Distribution sigma_dp [e-4]',sigma_dp*1e4)
sigma_dp=2e-4 
print('Used sigma_dp [e-4]',sigma_dp*1e4)

############################################################


##################
# Laser Cooler   #
##################

#laser-ion beam collision angle
theta_l = 2.6*np.pi/180 # rad
#theta_l = ion.theta_l
nx = 0; 
ny = -np.sin(theta_l); 
nz = -np.cos(theta_l)

# Ion excitation energy:
ion_excited_lifetime=ion.excited_lifetime
hw0 = ion.hw0 # eV
hc=constants.hbar*clight/constants.e # eV*m (ħc)
lambda_0 = 2*np.pi*hc/hw0 # m -- ion excitation wavelength

# Shift laser wavelength for fast longitudinal cooling:
#lambda_l = lambda_l*(1+1*sigma_dp) # m
lambda_l = ion.lambda_l

laser_frequency = clight/lambda_l # Hz
sigma_w = 2*np.pi*laser_frequency*sigma_dp
#sigma_w = 2*np.pi*laser_frequency*sigma_dp/2 # for fast longitudinal cooling

sigma_t = 1/sigma_w # sec -- Fourier-limited laser pulse
print('Laser pulse duration sigma_t = %.2f ps' % (sigma_t/1e-12))
print('Laser wavelength = %.2f nm' % (lambda_l/1e-9))

laser_waist_radius = 1.3e-3 #m
laser_energy = 5e-3
#laser_energy = ion.pulse_energy   
print('Laser pulse duration sigma_t = %.2f ps' % (sigma_t/1e-12))

laser_x = 0.5033557046979871*1e-3

GF_IP = xt.PulsedLaser(
                laser_x               = laser_x,
                laser_y               = 0,
                laser_z               = 0,
                laser_direction_nx    = 0,
                laser_direction_ny    = ny,
                laser_direction_nz    = nz,
                laser_energy          = laser_energy, # J
                laser_duration_sigma  = sigma_t, # sec
                laser_wavelength      = lambda_l, # m
                laser_waist_radius    = laser_waist_radius, # m
                laser_waist_shift     = 0, # m
                ion_excitation_energy = hw0, # eV
                ion_excited_lifetime  = ion_excited_lifetime, # sec                   
                )

##################
# MONITOR        #
##################
# create a monitor object, to reduce holded data
monitor = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=1,
                        n_repetitions=int(num_turns/save_interval),
                        repetition_period=save_interval,
                        num_particles=n_part)



#################################################
# Assemble full machine and initiliase tracker  #
#################################################
line = xt.Line(
        elements=[monitor,GF_IP,arc])
context = xo.ContextCpu(omp_num_threads='auto')
line.build_tracker(_context=context)



##### ACTUAL TRACKING #####
#  track all particles, *without* keeping turn-by-turn data (too memory expensive!)
line.track(particles, num_turns=num_turns,
        turn_by_turn_monitor=False,with_progress=True)


##### DATA ANALYSIS AND PLOT #####
# extract relevant values from monitor object
#  [all times, all particles, first (and only) monitor]
out_x     = monitor.x[:,:,0]
out_px    = monitor.px[:,:,0]
out_y     = monitor.y[:,:,0]
out_py    = monitor.py[:,:,0]
out_delta = monitor.delta[:,:,0]
out_state = monitor.state[:,:,0]
out_time  = monitor.at_turn[:, 0, 0] * T_per_turn

# find excited particles at each time step, and fraction of them....
out_excited      = out_state==2
out_excited_frac = 100*np.sum(out_excited, axis=1)/n_part


# compute single particle actions:
#   - for x, remove the dp/p contribution before computing
#   - for y, simple compututation:
gamma_x = (1+alpha_x**2)/beta_x
out_action_x = (gamma_x*(out_x-Dx*out_delta)**2 + 2*alpha_x*(out_x-Dx*out_delta)*(out_px-Dpx*out_delta) + beta_x*(out_px-Dpx*out_delta)**2)/2
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


###################################
##### produce some plots      #####
###################################

###################################
# plot excited particle fraction as a function of time
plt.figure(figsize=(8,6))       
plt.plot(out_time, out_excited_frac)
plt.xlabel('Time [s]')
plt.ylabel('Excited particles [%]')
plt.show()

###################################
# plot rms emittances and dp/p
plt.figure(figsize=(10,8))
plt.subplot(2,1,1)
plt.plot(out_time, out_emittance_x*1e6, label=f'$\epsilon_x$ (init={gemitt*1e6})')
plt.plot(out_time, out_emittance_y*1e6, label=f'$\epsilon_y$ (init={gemitt*1e6})')
plt.ylim([0, gemitt*1e6*1.05])
plt.ylabel('RMS $\epsilon$ [$\mu$m]')
plt.xlabel('Time [s]')
plt.legend()

plt.subplot(2,1,2)
plt.plot(out_time, out_sigma_p*1e4, label=f'RMS dp/p (init={sigma_dp*1e4})')
plt.ylim([0, sigma_dp*1e4*1.05])
plt.ylabel('RMS dp/p [$10^{-4}$]')
plt.xlabel('Time [s]')
plt.legend()

plt.show()

###################################
# plot emittances and dp/p using 95% definition
plt.figure(figsize=(10,8))

plt.subplot(2,1,1)
plt.plot(out_time, out_emittance_x_95*1e6, label=f'$\epsilon_x$')
plt.plot(out_time, out_emittance_y_95*1e6, label=f'$\epsilon_y$')
plt.ylim([0, 3*gemitt*1e6*1.05])
plt.ylabel('95% $\epsilon$ [$\mu$m]')
plt.xlabel('Time [s]')
plt.legend()

plt.subplot(2,1,2)
plt.plot(out_time, out_sigma_p_95*1e4, label=f'dp/p')
plt.ylim([0, 2*sigma_dp*1e4*1.05])
plt.ylabel('95% dp/p [$10^{-4}$]')
plt.xlabel('Time [s]')
plt.legend()

plt.show()

###################################
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

###################################
# Full phase space plot (x vs delta) at given time step, with histograms
# Specify which time step to plot:
idx_to_plot = -1 # last time step
idx_to_plot =  1 # first time step (after first turn/excitation)

# Create figure with slightly relaxed spacing
fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(4, 4, wspace=0.7, hspace=0.7)

# Define subplots
ax_main   = fig.add_subplot(gs[1:4, 0:3])
ax_top    = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
ax_right  = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

# -----------------------------------------------------
# 1. Main scatter plot: show all particles and excited particles
# -----------------------------------------------------
# Multiply x by 1000 to convert to mm
ax_main.scatter(out_x[idx_to_plot,:] * 1e3, out_delta[idx_to_plot,:], 
                s=10, alpha=0.4, label='All Particles')
ax_main.scatter(out_x[idx_to_plot,out_excited[idx_to_plot, :]] * 1e3, out_delta[idx_to_plot,out_excited[idx_to_plot, :]], 
                s=10, color='orange', label='Excited Particles')
ax_main.set_xlabel('x [mm]', labelpad=8)
ax_main.set_ylabel('Δp/p', labelpad=8)
ax_main.legend(loc='upper right')

# -----------------------------------------------------
# 2. Format axes: Using MaxNLocator and scientific notation
# -----------------------------------------------------
ax_main.xaxis.set_major_locator(  MaxNLocator(5) )
ax_main.yaxis.set_major_locator(  MaxNLocator(5) )
ax_top.xaxis.set_major_locator(   MaxNLocator(5) )
ax_right.yaxis.set_major_locator( MaxNLocator(5) )

ax_main.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
ax_main.tick_params(  axis='both', which='major', pad=8)
ax_top.tick_params(   axis='both', which='major', pad=8)
ax_right.tick_params( axis='both', which='major', pad=8)

# -----------------------------------------------------
# 3. Top histogram: x distribution for excited particles only
# -----------------------------------------------------
bins_x = 30
ax_top.hist(out_x[idx_to_plot,out_excited[idx_to_plot, :]] * 1e3, bins=bins_x, alpha=0.5, color='orange')
ax_top.set_ylabel('Count', labelpad=8)

# -----------------------------------------------------
# 4. Right histogram: Δp/p distribution for excited particles only
# -----------------------------------------------------
bins_delta = 30
ax_right.hist(out_delta[idx_to_plot,out_excited[idx_to_plot, :]], bins=bins_delta, orientation='horizontal',
              alpha=0.5, color='orange')
ax_right.set_xlabel('Count', labelpad=8)
plt.show()

###### END OF SCRIPT ######