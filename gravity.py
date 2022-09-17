#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 15:23:10 2021

CompMod Ex. 3: Functions describing the interaction of two
particles via gravity. The functions take two Particle3D
instances and the potential parameters as arguments and return the potential
energy of the particles and the force between them.

@author: Mika Kontiainen, s1853201
"""

import numpy as np
from particle3D import Particle3D as p3d


def pot_energy_gravity(particle_1, particle_2):
    """
    Returns the potential energy of two Particle3D instances
    interacting via gravity
    
    :param particle_1: Particle3D instance.
    :param particle_2: Particle3D instance.
    :param *args: List containing the Morse potential parameters D_e, r_e and
                  alpha controlling the depth, position and curvature of the
                  potential respectively.
        
    :return U_m: float, gravitational potential energy of the two particles.
    """
    
    #Gravitational constant
    #G = 4.30091E-3 #(parsec x solar mass^-1 (km/s)^2)
    G = 1
        
    #Compute separation between the two particles
    r_sep = np.linalg.norm(particle_2.pos - particle_1.pos)
    
    #Compute the potential energy
    U_g = -G * particle_1.mass * particle_2.mass / r_sep
    
    return U_g


def force_gravity(particle_1, particle_2):
    """
    Returns the force between two Particle3D instances
    interacting via gravity
    
    :param particle_1: Particle3D instance.
    :param particle_2: Particle3D instance.
    :param *args: List containing the Morse potential parameters D_e, r_e and
                  alpha controlling the depth, position and curvature of the
                  potential respectively.
    
    :return force_12: [3] array w/ force on particle 1 due to particle 2.
    """
    
    #Gravitational constant
    #G = 4.30091E-3 #(parsec x solar mass^-1 (km/s)^2)
    G = 1
        
    #Compute separation between the two particles
    r_sep = np.linalg.norm(particle_2.pos - particle_1.pos)
    
    #Compute the separation unit vector r_hat
    r_hat = (particle_2.pos - particle_1.pos) / r_sep
    
    #Compute the force vector on particle 1 due to particle 2
    force_12 = G * particle_1.mass * particle_2.mass / r_sep**2 * r_hat
        
    return force_12, r_sep