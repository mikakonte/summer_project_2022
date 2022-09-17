#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 13:09:20 2021

CompMod Ex. 2: Particle3D, a class to describe classical point particles in
3D space.

An instance of the class describes a particle of a given mass in Euclidean 3D
space with position and velocity represented by NumPy arrays of dimension [3].

Includes 1st and 2nd order time integrator methods for updating the position
and a 1st order method for updating the velocity as well as a method for
creating a new instance of Particle3D from an open text file.

@author: Mika Kontiainen, s1853201
"""

import numpy as np

class Particle3D(object):
    """
    Class to describe classical point particles in 3D space

        Properties:
    label: name of the particle
    mass: mass of the particle
    pos: position of the particle
    vel: velocity of the particle

        Methods:
    __init__ - initialises the particle in 3D space
    __str__ - defines print output in standard XYZ format
    kinetic_e - computes the kinetic energy
    momentum - computes the linear momentum
    update_pos - updates the position to 1st order
    update_pos_2nd - updates the position to 2nd order
    update_vel - updates the velocity

        Static Methods:
    new_p3d - initialises a P3D instance from a file handle
    sys_kinetic - computes total K.E. of a p3d list
    com_velocity - computes total mass and CoM velocity of a p3d list
    """
    
    def __init__(self, label, mass, position, velocity):
        """
        Initialises a particle in 3D space

        :param label: String w/ the name of the particle
        :param mass: float, mass of the particle
        :param position: [3] float array w/ position
        :param velocity: [3] float array w/ velocity
        """
        self.label = label
        self.mass = mass
        self.pos = np.array(position)
        self.vel = np.array(velocity)
        
    def __str__(self):
        """
        Returns an XYZ-compliant string. The format is
        <label>    <x>  <y>  <z>
        """
        return str("{}   {} {} {}".format(self.label, self.pos[0], \
                                        self.pos[1], self.pos[2]))
    
    def kinetic_e(self):
        """
        Returns the kinetic energy of a Particle3D instance

        :return ke: float, 1/2 m v**2
        """
        return float(0.5 * self.mass * np.linalg.norm(self.vel)**2)
    
    def momentum(self):
        """
        Returns the momentum of a Particle3D instance
        
        :return momentum: float, m v
        """
        return np.array(self.mass * self.vel)
    
    def update_pos(self, dt):
        """
        Updates the position of the Particle3D instance to first order
        
        :param dt: float, time step for integration
        """
        self.pos = self.pos + float(dt) * self.vel
        
    def update_pos_2nd(self, dt, force):
        """
        Updates the position of the Particle3D instance to second order
        
        :param dt: float, time step for integration
        :param force: [3] float array w/ force
        """
        self.pos = self.pos + float(dt) * self.vel + \
            float(dt)**2 * np.array(force) / (2 * self.mass)
            
    def update_vel(self, dt, force):
        """
        Updates the velocity of the Particle3D instance
        
        :param dt: float, time step for integration
        :param force: [3] float array w/ force
        """
        self.vel = self.vel + float(dt) * np.array(force) / self.mass
        
    @staticmethod
    def new_p3d(file_handle):
        """
        Initialises a Particle3D instance given an input file handle.
        
        The input file should contain one line per particle in the format:
        label   <mass>  <x> <y> <z>    <vx> <vy> <vz>
        
        :param inputFile: Readable file handle in the above format

        :return Particle3D instance
        """
        
        #Read particle parameters into a list
        attr = file_handle.readline().split()
        
        err_mes = "Please check the data is in the format: \
            label   <mass>   <x> <y> <z>   <vx> <vy> <vz>"
        if len(attr) == 8:
            try:
                #Convert numerical parameters into floats
                attr[1:8] = [float(n) for n in attr[1:8]]
                label = str(attr[0])
                mass = float(attr[1])
                pos = np.array(attr[2:5])
                vel = np.array(attr[5:8])
            except:
                #Return error if parameters cannot be converted into floats
                print('bing')
                print(err_mes)
                return
        else:
            #Return error if the input is of a wrong length
            print('bong')
            print(err_mes)
            return
        
        return Particle3D(label, mass, pos, vel)
    
    @staticmethod
    def sys_kinetic(p3d_list):
        """
        Computes the total kinetic energy of a list of P3D's'
        
        :param p3d_list: list in which each item is a P3D instance
        :return total_ke: The total kinetic energy of the system
        """
        total_ke = 0.0
        
        for p in p3d_list:
            #Add individual ke contributions to the total
            total_ke += p.kinetic_e()
            
        return total_ke
    
    @staticmethod
    def sys_energy(p3d_list, pot_func):
        total_energy = 0.0
        kinetic = 0.0
        potential = 0.0
        for n in range(len(p3d_list)):
            #Add individual ke contributions to the total
            kinetic += p3d_list[n].kinetic_e()
            for m in range(n+1, len(p3d_list)):
                potential += pot_func(p3d_list[n], p3d_list[m])
        total_energy = kinetic + potential
        return total_energy, kinetic, potential
    
    @staticmethod
    def com_velocity(p3d_list):
        """
        Computes the total mass and CoM velocity of a list of P3D's

        :param p3d_list: list in which each item is a P3D instance
        :return total_mass: The total mass of the system 
        :return com_vel: Centre-of-mass velocity
        """
        total_mass = 0.0
        com_vel = np.zeros(3)
        
        for p in p3d_list:
            #Add individual mass and com_vel contributions to the totals
            total_mass += p.mass
            com_vel += p.mass * p.vel
            
        com_vel = com_vel / total_mass
        
        return total_mass, com_vel
    
    @staticmethod
    def sys_angular(p3d_list):
        """
        Computes the angular momentum about the CoM of the system

        :param p3d_list: list in which each item is a P3D instance
        :return com: centre-of-mass of the system
        :return angular: The total angular momentum of the system
        """
        total_mass = sum([p.mass for p in p3d_list])
        com = np.array(sum([p.mass * p.pos / total_mass for p in p3d_list]))
        angular = np.zeros(3)
        
        for p in p3d_list:
            r_com = p.pos - com
            angular += np.cross(p.vel, r_com)
            
        return com, np.linalg.norm(angular)
    
    @staticmethod
    def get_forces(p3d_list, force_func, *args):
        """
        Computes and returns the interparticle force matrix for a given function.
        
        :param p3d_list: list in which each item is a P3D instance
        :param force_func: function to return the force on P3D 1 due to P3D 2
        :*args: additional arguments for force_func
        :return force: array of the interparticle forces
        """
        #Define list for storing r_min / homology radius
        dist = []
        
        force = np.empty((len(p3d_list), len(p3d_list)), dtype = object)
        for n in range(len(p3d_list)):
            force[n, n] = np.zeros(3)
            for m in range(n+1, len(p3d_list)):
                force_nm, r_sep = force_func(p3d_list[n], p3d_list[m], *args)
                #Use symmetry to determine forces between particles n, m
                force[n, m], force[m, n] = force_nm, -force_nm
                #Set particle self-interaction to zero
                force[m, m] = np.zeros(3)
                
                dist.append(r_sep)
                
        minmax_ratio = min(dist) / max(dist)
        
        return force, minmax_ratio
    
    @staticmethod
    def force(p3d_list, force_func):
        forces = np.zeros((len(p3d_list), 3))
        for n in range(len(p3d_list)):
            for m in range(n+1, len(p3d_list)):
                force_nm, r_sep = force_func(p3d_list[n], p3d_list[m])
                forces[n] += force_nm
                forces[m] -= force_nm
        return forces

    
    @staticmethod
    def get_omega(p3d_list):
        #Used for time-transformed leapfrog
        omega = 0
        G = 3*[np.zeros(3)]
        
        for k in range(len(p3d_list)):
            for j in range(k+1, len(p3d_list)):
                #Product of masses
                mu = p3d_list[k].mass * p3d_list[j].mass
                #Particle separation
                r_sep = np.linalg.norm(p3d_list[k].pos - p3d_list[j].pos)
                #Update omega and its derivative (Aarseth 5.9)
                G[k] = G[k] + (p3d_list[j].pos - p3d_list[k].pos) / r_sep**3
                G[j] = G[j] + (p3d_list[k].pos - p3d_list[j].pos) / r_sep**3
                omega += mu /r_sep
                
        w = sum([np.dot(p3d_list[k].vel, G[k]) for k in range(len(p3d_list))]) / omega
                
        return omega, abs(w)
