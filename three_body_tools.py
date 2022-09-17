#MODULE FOR STUDYING THE 3BP

import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from particle3D import Particle3D as p3d
from IPython import display
import copy
import collections
import argparse
from gravity import *
from load_xyz import *


#Loads input xyz file and returns a list of particles
def file_loader(input_file):
    '''
    Input:
    input_file – name of input file
    Output:
    p – particle list
    '''
    infile = open(input_file, 'r')
    #Obtain particle values from input file
    num_lines = sum(1 for line in infile)
    infile.seek(0)
    infile.readline()
    infile.readline()
    p = []
    for n in range(num_lines - 2):
        p.append(p3d.new_p3d(infile))
    #Close input file
    infile.close()
    return p


#Saves coordinates into a file
def file_saver(output_file, p, pos, vel, t):
    '''
    Input:
    output_file – name of output file
    p – particle list
    pos – array of position coordinates
    vel – array of velocity coordinates
    t – array of time steps
    Output:
    Saves coordinates into a file.
    '''
    outfile = open(output_file, "w")
    #Save data into the output file // every 5th data point
    for i in range(0, len(t), 5):
        outfile.write(f'{len(p)}\n')
        outfile.write(f'{t[i]}\n')
        #outfile.write(f'Point = {i}\n')
        for n in range(len(p)):
            outfile.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'.format(
                p[n].label, p[n].mass, pos[i][n, 0], pos[i][n, 1], pos[i][n, 2], vel[i][n, 0], vel[i][n, 1], vel[i][n, 2]))
    #Close output file
    outfile.close()
    return


#Simplified velocity Verlet for use with main integrator
def velocity_verlet(p, force_func, pot_func, dur, dt, *args):
    """
    Input:
    p – list of Particle3D instances
    force_func – function to return force between two particles
    pot_fun – function to return potential energy between two particles
    dur – total duration of the integration
    dt – time-step used for the integration
    args – additional arguments for the force and energy function
    Output:
    p – updated list of Particle3D instances
    """
    #Correct for centre-of-mass drift
    for particle in p:
        particle.vel -= particle.com_velocity(p)[1]
    #Set initial time
    t = 0
    #Initialise lists for holding separation and total energy data
    time = [t]
    pos = [[p[n].pos] for n in range(len(p))]
    #Obtain initial force
    force, r_min = p[0].get_forces(p, force_func)
    #Iterate over total integration time t in steps of length dt
    while t <= dur:
        #UPDATE PARTICLE POSITIONS
        for n in range(len(p)):
            p[n].update_pos_2nd(dt, np.sum(force[n, :]))    
        #Compute the new interparticle force matrix
        force_new, r_min = p[0].get_forces(p, force_func)
        #UPDATE PARTICLE VELOCITIES
        for n in range(len(p)):
            #Sum over force contributions on particle n
            p[n].update_vel(dt, 0.5*np.sum(force[n, :] + force_new[n, :]))
        #Redefine the force matrix for the next step
        force = force_new
        #Save time and particle positions
        t += dt
        time.append(t)
        for n in range(len(p)):
            pos[n].append(p[n].pos)
    #Return particle list
    return p


#Main integrator with variable time step
def verlet(input_file, output_file, dur, dt, tol):
    '''
    Input:
    input_file – name of input file
    output_file – name of output file
    dur – total duration of the integration
    dt – time-step used for the integration
    tol – energy error tolerance
    Output:
    pos – position array
    vel – velocity array
    t – time array
    Prints energy variation.
    '''
    p = file_loader(input_file)
    t = [0]
    pos = [[p[n].pos for n in range(len(p))]]
    vel = [[p[n].vel for n in range(len(p))]]
    E_0, E_k, E_p = p[0].sys_energy(p, pot_energy_gravity)
    delta = dt
    while t[-1] < dur:
        p = velocity_verlet(p, force_gravity, pot_energy_gravity, delta, delta)
        t.append(t[-1]+delta)
        pos.append([p[n].pos for n in range(len(p))])
        vel.append([p[n].vel for n in range(len(p))])
        E, E_k, E_p = p[0].sys_energy(p, pot_energy_gravity)
        if abs(E/E_0) <= 1+tol:
            delta = abs(E_0/E_p)**2 * dt
        else:
            delta = abs(E_0/E_p)**1 * dt
    print(abs((abs(E) - abs(E_0))/E_0))
    pos, vel, t = np.array(pos), np.array(vel), np.array(t)
    file_saver(output_file, p, pos, vel, t)
    return pos, vel, t


#Function to find index of ejected body at each time step
def find_eject(xyz_data):
    '''
    Input:
    xyz_data = xyz data handle of .dat file loaded using load_xyz.py
    Output:
    eject – index of ejected body at each time step
    '''
    #Find third (non-binary) body using average position
    star_pos = np.array([np.array(xyz_data['Star_A|0']), np.array(xyz_data['Star_B|1']), np.array(xyz_data['Star_C|2'])])
    eject = np.empty(len(star_pos[0]), dtype = int)
    for i in range(len(star_pos[0])):
        avg_pos = np.array(sum([star_pos[n][i,1:4] for n in range(len(star_pos))])/len(star_pos))
        dist = np.array([np.linalg.norm(star_pos[n][i,1:4] - avg_pos) for n in range(len(star_pos))])
        eject[i] = dist.argmax()
    return eject


#Calculate Shannon entropy at all time steps using discretised interpolation function
def find_shannon(xyz_data, time):
    '''
    Input:
    xyz_data – xyz data handle of .dat file loaded using load_xyz.py
    time – array of required sampling times
    Return:
    shannon_entropy – value of the Shannon entropy of the system at times given in time
    eject_even – evenly sampled ejection data
    '''
    shannon_entropy = np.empty(len(time))
    #Find ejected body at all time steps
    eject = find_eject(xyz_data)
    #Form interpolation function of ejects for even sampling
    eject_func = interpolate.interp1d(xyz_data['Star_A|0'][:,0], eject, kind = 'linear')
    #Save evenly sampled ejects for calculating entropy
    eject_even = eject_func(time)
    #Loop through all time steps
    for step in range(1, len(time)+1):
        slice = eject_even[0:step]
        #Calculate the number of 0, 1, 2 occurrences in eject
        occur = [np.count_nonzero(slice == n) for n in range(3)]
        #Calculate Shannon entropy
        shannon = -sum([occur[i]/len(slice) * np.log2(occur[i]/len(slice)) for i in range(3) if occur[i] > 0])
        shannon_entropy[step-1] = shannon
    return shannon_entropy, eject_even


#Time-transformed leapfrog (Mikkola & Tanikawa, 1999)
def ttl(input_file, output_file, dur, dt, particle_input=False):
    '''
    Input:
    input_file – name of input file
    output_file – name of output file
    dur – total duration of the integration
    dt – time-step used for the integration
    particle_input – input_file interpreted as a particle list (True/false)
    Output:
    pos – position array
    vel – velocity array
    t – time array
    Prints energy variation.
    '''
    if particle_input is True:
        p = input_file
    else:
        p = file_loader(input_file)
    t = [0]
    pos = [[p[n].pos for n in range(len(p))]]
    vel = [[p[n].vel for n in range(len(p))]]
    E_0, T, U = p[0].sys_energy(p, pot_energy_gravity)
    P_t = -T-U
    while t[-1] <= dur:
        tempo = t[-1]
        #DRIFT
        for k in range(len(p)):
            p[k].update_pos(dt/(2*(T+P_t)))
        #FORWARD TIME
        tempo += dt/(2*(T+P_t))
        #GET NEW ENERGY AND FORCE
        E, T, U = p[0].sys_energy(p, pot_energy_gravity)
        force = p[0].force(p, force_gravity)
        #KICK
        for k in range(len(p)):
            p[k].update_vel(dt/(-U), force[k])
        #GET NEW ENERGY
        E, T, U = p[0].sys_energy(p, pot_energy_gravity)
        #DRIFT
        for k in range(len(p)):
            p[k].update_pos(dt/(2*(T+P_t)))
        #FORWARD TIME
        tempo += dt/(2*(T+P_t))
        #SAVE VALUES
        t.append(tempo)
        pos.append([p[n].pos for n in range(len(p))])
        vel.append([p[n].vel for n in range(len(p))])
    print(abs((abs(E) - abs(E_0))/E_0))
    pos, vel, t = np.array(pos), np.array(vel), np.array(t)
    file_saver(output_file, p, pos, vel, t)
    return pos, vel, t


#Compute divergence between two solutions
def divergence(sol1, sol2, time):
    '''
    Input:
    sol1 – solution 1 xyz data handle of .dat file loaded using load_xyz.py
    sol2 – solution 2 xyz data handle of .dat file loaded using load_xyz.py
    time – array of required sampling times
    Output:
    div – time series of position divergence
    '''
    div = np.zeros(len(time))
    labels = ['A', 'B', 'C']
    for n in range(3):
        for i in range(3):
            mass1 = interpolate.interp1d(sol1[f'Star_{labels[n]}|{n}'][:,0], \
                                         sol1[f'Star_{labels[n]}|{n}'][:,i+1], kind = 'cubic')
            mass2 = interpolate.interp1d(sol2[f'Star_{labels[n]}|{n}'][:,0], \
                                         sol2[f'Star_{labels[n]}|{n}'][:,i+1], kind = 'cubic')
            div += (mass1(time)-mass2(time))**2
    div = np.sqrt(div)
    return div

#Compute divergence between two solutions
def divergence_ps(sol1, sol2, time):
    '''
    Input:
    sol1 – solution 1 xyz data handle of .dat file loaded using load_xyz.py
    sol2 – solution 2 xyz data handle of .dat file loaded using load_xyz.py
    time – array of required sampling times
    Output:
    div – time series of position divergence
    '''
    div = np.zeros(len(time))
    labels = ['A', 'B', 'C']
    for n in range(3):
        for i in range(6):
            mass1 = interpolate.interp1d(sol1[f'Star_{labels[n]}|{n}'][:,0], \
                                         sol1[f'Star_{labels[n]}|{n}'][:,i+1], kind = 'cubic')
            mass2 = interpolate.interp1d(sol2[f'Star_{labels[n]}|{n}'][:,0], \
                                         sol2[f'Star_{labels[n]}|{n}'][:,i+1], kind = 'cubic')
            div += (mass1(time)-mass2(time))**2
    div = np.sqrt(div)
    return div

#Compute the homology radius
def homology(xyz_data):
    '''
    Input:
    xyz_data – xyz data handle of .dat file loaded using load_xyz.py
    Output:
    r_hom – time series of homology radius r_min/r_max
    r_avg – time series of average separation
    '''
    star_pos = np.array([np.array(xyz_data['Star_A|0']), np.array(xyz_data['Star_B|1']), np.array(xyz_data['Star_C|2'])])
    r_hom = np.empty(len(star_pos[0]), dtype = float)
    r_avg = np.empty(len(star_pos[0]), dtype = float)
    for i in range(len(star_pos[0])):
        r_sep = np.array([np.linalg.norm(star_pos[n-1][i,1:4] - star_pos[n][i,1:4]) for n in range(len(star_pos))])
        r_hom[i] = min(r_sep)/max(r_sep)
        r_avg[i] = np.average(r_sep)
    return r_hom, r_avg

#Compute homology radius and angle
def homology_param(xyz_data):
    '''
    Input:
    xyz_data – xyz data handle of .dat file loaded using load_xyz.py
    Output:
    r_hom – time series of homology radius r_min/r_max
    r_avg – time series of average separation
    '''
    star_pos = np.array([np.array(xyz_data['Star_A|0']), np.array(xyz_data['Star_B|1']), np.array(xyz_data['Star_C|2'])])
    r_hom = np.empty(len(star_pos[0]), dtype = float)
    angle = np.empty(len(star_pos[0]), dtype = float)
    for i in range(len(star_pos[0])):
        r_sep = np.array([np.linalg.norm(star_pos[n-1][i,1:4] - star_pos[n][i,1:4]) for n in range(len(star_pos))])
        r_hom[i] = min(r_sep)/max(r_sep)
        angle[i] = np.arccos((max(r_sep)**2 + min(r_sep)**2 - np.median(r_sep)**2)/(2*max(r_sep)*min(r_sep)))
    return r_hom, angle

#Determine which region of the homology map the solution is in
def find_region(r_hom, angle):
    '''
    Input:
    r_hom – time series of homology radius from homology_param
    angle – time series of homology angle from homology_param
    Output:
    region – time series of currently occupied region symbolised as (0123) for (HAML)
    '''
    region = np.empty(len(r_hom))
    for i in range(len(r_hom)):
        if r_hom[i] <= 1/3:
            region[i] = 0 #Hierarchical
        elif r_hom[i] >= 2/3:
            region[i] = 3 #Lagrangian
        else:
            if angle[i] >= np.arctan(1/2):
                region[i] = 2 #Middle
            else:
                region[i] = 1 #Aligned
    return region

#Calculate Kolmogorov-Sinai / Shannon entropy at all time steps using discretised interpolation function
def ks_entropy(region, sol_time, time):
    '''
    Input:
    region – time series of homology map region from find_region
    sol_time – original time series of the solution
    time – array of required sampling times
    Return:
    ks – value of the KS entropy of the system at times given in time
    region_even – evenly sampled region data
    '''
    ks = np.empty(len(time))
    #Form interpolation function of regions for even sampling
    region_func = interpolate.interp1d(sol_time, region, kind = 'linear')
    #Save evenly sampled regions for calculating entropy
    region_even = region_func(time)
    #Loop through all time steps
    for step in range(1, len(time)+1):
        slice = region_even[0:step]
        #Calculate the number of 0, 1, 2, 3 occurrences in region
        occur = [np.count_nonzero(slice == n) for n in range(4)]
        #Calculate Shannon entropy
        ks_unit = -sum([occur[i]/len(slice) * np.log2(occur[i]/len(slice)) for i in range(4) if occur[i] > 0])
        ks[step-1] = ks_unit
    return ks, region_even

#Compute entropy based on a partitioned homology radius
def homology_entropy(r_hom, sol_time, time, partition):
    '''
    Input:
    r_hom – time series of homology radius from homology_param
    sol_time – original time series of the solution
    time – array of required sampling times
    partition – number of partitions
    Return:
    entropy – entropy of the system at times given in time
    '''
    region = np.empty(len(r_hom))
    entropy = np.empty(len(time))
    for i in range(len(r_hom)):
        region[i] = np.floor(partition*r_hom[i])
    #Form interpolation function of regions for even sampling
    region_func = interpolate.interp1d(sol_time, region, kind = 'linear')
    #Save evenly sampled regions for calculating entropy
    region_even = region_func(time)
    #Loop through all time steps
    for step in range(1, len(time)+1):
        slice = region_even[0:step]
        #Calculate the number of occurrences in region
        occur = [np.count_nonzero(slice == n) for n in range(partition)]
        #Calculate Shannon entropy
        entropy_unit = -sum([occur[i]/len(slice) * np.log2(occur[i]/len(slice)) for i in range(partition) if occur[i] > 0])
        entropy[step-1] = entropy_unit
    return entropy, region_even

#Obtain potential and kinetic energy of the ejected body
def energies(xyz_data):
    '''
    Input:
    xyz_data – xyz data handle of .dat file loaded using load_xyz.py
    Output:
    pe – potential energy of the ejected body
    ke – kinetic energy of the ejected body
    '''
    #Set value of gravitational constant
    G = 1
    eject = find_eject(xyz_data)
    coordinates = np.array([np.array(xyz_data['Star_A|0']),\
                            np.array(xyz_data['Star_B|1']), np.array(xyz_data['Star_C|2'])])
    #Create empty lists
    pe = np.empty(len(coordinates[0]), dtype = float)
    ke = np.empty(len(coordinates[0]), dtype = float)
    for i in range(len(coordinates[0])):
        #Obtain potential energy
        separations = np.array([np.linalg.norm(coordinates[n][i,1:4] - coordinates[eject[i]][i,1:4])\
                                for n in range(len(coordinates))])
        pe[i] = 0.5*sum([-G*0.5*0.5/separations[n] for n in range(len(coordinates)) if separations[n] > 0])
        #Obtain kinetic energy
        ke[i] = 0.5*0.5*np.linalg.norm(coordinates[eject[i]][i,4:7])**2
    return pe, ke

#Compute individual divergence rates between two solutions in position space
def divergence_ind(sol1, sol2, time):
    '''
    Input:
    sol1 – solution 1 xyz data handle of .dat file loaded using load_xyz.py
    sol2 – solution 2 xyz data handle of .dat file loaded using load_xyz.py
    time – array of required sampling times
    Output:
    div – array of three time series of position divergence
    '''
    div_list = np.array(3*[np.zeros(len(time))])
    labels = ['A', 'B', 'C']
    for n in range(3):
        div = 0
        for i in range(3):
            mass1 = interpolate.interp1d(sol1[f'Star_{labels[n]}|{n}'][:,0], \
                                         sol1[f'Star_{labels[n]}|{n}'][:,i+1], kind = 'cubic')
            mass2 = interpolate.interp1d(sol2[f'Star_{labels[n]}|{n}'][:,0], \
                                         sol2[f'Star_{labels[n]}|{n}'][:,i+1], kind = 'cubic')
            div += (mass1(time)-mass2(time))**2
        div_list[n] = np.sqrt(div)
    return div_list

#Compute individual divergence rates between two solutions in phase space
def divergence_ind_ps(sol1, sol2, time):
    '''
    Input:
    sol1 – solution 1 xyz data handle of .dat file loaded using load_xyz.py
    sol2 – solution 2 xyz data handle of .dat file loaded using load_xyz.py
    time – array of required sampling times
    Output:
    div – array of three time series of position divergence
    '''
    div_list = np.array(3*[np.zeros(len(time))])
    labels = ['A', 'B', 'C']
    for n in range(3):
        div = 0
        for i in range(6):
            mass1 = interpolate.interp1d(sol1[f'Star_{labels[n]}|{n}'][:,0], \
                                         sol1[f'Star_{labels[n]}|{n}'][:,i+1], kind = 'cubic')
            mass2 = interpolate.interp1d(sol2[f'Star_{labels[n]}|{n}'][:,0], \
                                         sol2[f'Star_{labels[n]}|{n}'][:,i+1], kind = 'cubic')
            div += (mass1(time)-mass2(time))**2
        div_list[n] = np.sqrt(div)
    return div_list

#Obtain the total energy of the equal-mass system
def total_energy(xyz_data, masses = [0.5, 0.5, 0.5]):
    '''
    Input:
    xyz_data – xyz data handle of .dat file loaded using load_xyz.py
    Output:
    energy – time series of the energy of the system
    '''
    #Set value of gravitational constant
    G = 1
    coordinates = np.array([np.array(xyz_data['Star_A|0']),\
                            np.array(xyz_data['Star_B|1']), np.array(xyz_data['Star_C|2'])])
    #Create empty lists
    pe = np.empty(len(coordinates[0]), dtype = float)
    ke = np.empty(len(coordinates[0]), dtype = float)
    for i in range(len(coordinates[0])):
        #Obtain potential energy
        separations = np.array([np.linalg.norm(coordinates[n-1][i,1:4] - coordinates[n][i,1:4])\
                                for n in range(len(coordinates))])
        pe[i] = sum([-G*masses[n-1]*masses[n]/separations[n] for n in range(len(coordinates)) if separations[n] > 0])
        #Obtain kinetic energy
        ke[i] = 0.5*sum(masses[n]*np.linalg.norm(coordinates[n][i,4:7])**2 for n in range(len(coordinates)))
    energy = ke + pe
    return energy