
from scipy.integrate import quad
import scipy.constants as const
import scipy
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import yaml


def main():
    #  PARSE
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        help='input file')
    input_file_name = parser.parse_args().input
    with open(input_file_name) as fid:
        my_input = yaml.load(fid,Loader=yaml.SafeLoader)

    #  WORK
    my_cq = CQ(my_input)
    my_cq.integrand()
    my_func = my_cq.fun
    my_dos = my_cq.dos
    #
    result = np.zeros(100)
    result_dos = np.zeros(100)
    Espace = np.linspace(-2,2,100)
    for i, energy in enumerate(Espace):
        result[i] = (my_func(energy*const.eV))
        result_dos[i] = my_dos(energy*const.eV)
    # plt.plot(Espace, result_dos)
    # plt.plot(Espace, result)
    # plt.grid()
    # plt.show()


    # integration
    n_EF = 100
    fermi_energy_range = np.linspace(-1, 1, n_EF)*const.eV
    cq = np.zeros(n_EF)
    for i, fermi_energy in enumerate(fermi_energy_range):
        my_cq.phi = fermi_energy_range[i]
        cq[i] = my_cq.compute_integral()


    print(cq)
#    plt.plot(fermi_energy_range/const.eV, cq/const.atto*const.micro)
    plt.grid()
    plt.xlabel('Energy, eV')
    plt.ylabel('Cq, aF/mkm')

    # Ci

    my_ci = Ci(my_input) # class instance
    intrinsic_capacitance = my_ci.compute_epsilon_i(my_input)
    my_ci.dummi_ci(my_input)
    print('empirical C = ', my_ci.empiric_ci)
#    plt.plot(fermi_energy_range/const.eV, my_ci.empiric_ci/const.atto*const.micro*np.ones(n_EF))
#    plt.show()
    plt.close()
    V = np.zeros(n_EF)

    coef = (my_ci.empiric_ci + cq)/my_ci.empiric_ci

    plt.figure()
    plt.plot(coef*fermi_energy_range/const.eV, fermi_energy_range/const.eV)
    plt.ylabel('phi')
    plt.xlabel('Voltage')
    plt.grid()
    plt.xlim(-1.5, 1.5)
    plt.show()
    print('I am done')


    #

class CQ:
    def __init__(self,input):
        self.input = input # all input available from everywhere
        self.a = 2.46 * const.angstrom  # Graphene Bravice lattice constant [m]
        self.gamma = 3.1 * const.eV  # graphene overlap [eV]
        self.g0 = 8 / np.sqrt(3) / self.a / np.pi / self.gamma  # prefactor density
        self.T = input['T']
        self.C1 = const.e*const.e/8/const.k/self.T
        self.E0 = input['E0']*const.eV
        self.E1 = input['E1']*const.eV
        self.EF = input['EF']*const.eV
        self.epsilon = input['epsilon']
        self.phi = input['phi']

    def g(self, E):
        E0 = self.E0
        E1 = self.E1
        g0 = self.g0
        first_mode = my_sqrt_m1(E**2 - E0**2)
        second_mode = my_sqrt_m1(E**2 - E1**2)

        g = g0/2*np.abs(E)*( first_mode + second_mode ) # density of states
        return g

    def fth(self, E):
        EF = self.EF
        kB = const.k
        T = self.T
        phi = self.phi
        ee = (E-EF+phi)/(2*kB*T)

        def sech(x):
            return 2.0/(np.exp(x) + np.exp(-x))
        return 1/const.k/T/4*sech(ee)*sech(ee)

    def integrand(self):
        e = const.e
        def func(E):
            g = self.g(E)
            fth = self.fth(E)
            return e**2 * g * fth
        self.fun = func  # function to integrate
        self.dos = self.g

    def compute_integral(self):
        #print('Fermi energy:', self.EF)
        I = quad(self.fun, -10.0*const.eV, 10*const.eV, epsabs=1e-14, points=[self.E0, self.E1, -self.E0, -self.E1])[0]
        print(I)
        pass
        return I

class Ci:
    def __init__(self, input):
        self.epsilon = input['epsilon']
        self.T = input['T']
        self.z = input['z']
        self.molar_weight = input['molar_weight']/const.kilo  # kg/mol
        self.m0 = self.molar_weight/const.N_A  # kg mass of the ion
        self.density = input['density']/const.kilo*100**(3)  #  kg/m^3
        self.n0 = self.density/self.m0 # no is the number of molecules per m**3
        self.l = np.sqrt(self.epsilon*const.k*self.T / self.z**2 * const.e**2 * self.n0)
        self.R = input['R']*const.nano # R?

    def compute_epsilon_i(self, input):
        print('2*l/gamma = ', 2*self.l/np.e)
        return 2*np.pi*const.epsilon_0*self.epsilon/np.log(2*self.l/self.R/np.e)

    def dummi_ci(self, input):
        self.l_empiric = 5*const.nano # 10 nm
        self.empiric_ci = 2*np.pi*self.epsilon*const.epsilon_0/np.log(2*self.l_empiric/self.R)

#def find_V_from_chi(Cq,Ci,):
#    def F(self,phi):
#        coef = /C2
#        return phi - 1/(1 + coef)*V


def my_sqrt_m1(x):
    if x >= 0:
        return 1.0/np.sqrt(x)
    else:
        return 0

if __name__ == '__main__':
    main()
