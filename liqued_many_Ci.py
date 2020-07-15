
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
########################################################################################################################
# Cq
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
#    plt.plot(Espace, result_dos)
#    plt.plot(Espace, result)
#    plt.ylabel('dos')
#    plt.xlabel('Energy, eV')
#    plt.grid()
#    plt.show()
#    plt.close()

    # integration
    n_phi = 100 # potential on a tube
    phi_min = -1.5
    phi_max = 1.5
    phi_range = np.linspace(phi_min, phi_max, n_phi)*const.eV
    cq = np.zeros(n_phi)
    for i, phi in enumerate(phi_range):
        my_cq.phi = phi  # change the attribute
        cq[i] = my_cq.compute_integral()

    print(cq)
    plt.plot(phi_range/const.eV, cq/const.atto*const.micro)
    plt.grid()
    plt.xlabel('$\phi$, eV')
    plt.ylabel('Cq, aF/mkm')
    plt.show()

########################################################################################################################
# Ci
    my_ci = Ci(my_input) # class instance
#    l_test = 10 #nm dipole layer length
#    my_ci.dummi_ci(l_test)  # with empirical dipole layer length
#    print('empirical C = ', my_ci.empiric_ci)
#    plt.plot(phi_range/const.eV, my_ci.empiric_ci/const.atto*const.micro*np.ones(n_EF))
#    plt.show()
#    plt.close()

    # n_ci number of different ci
    n_l = 10 # the same number of ci is expected
    l_min = 1 #nm
    l_max = 10 #nm
    l_range = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])*const.nano
#    l_range = np.linspace(l_min, l_max, n_l)*const.nano #m
    n_l = len(l_range)
    coef_many_ci = np.zeros([n_phi, n_l])
    ci_range = np.zeros(n_l)

    for i, l in enumerate(l_range):
        my_ci.dummi_ci(l) # my_ci.empiric_ci is assinged to a new value
        print('capacitance = ', my_ci.empiric_ci/const.atto*const.micro)
        ci_range[i] = my_ci.empiric_ci
        coef = (my_ci.empiric_ci + cq)/my_ci.empiric_ci # coef between phi and V VECTOR size: phi
        coef_many_ci[:,i] = coef

################################################# PLOTS #################################
    plot_phi_vs_Voltage(coef_many_ci,phi_range, l_range, my_input)
    plot_l_vs_ci(l_range, ci_range)
    test_K0_approx()
########################################################################################


########################################################################################################################
    Uds  =np.loadtxt('out/Uds.txt')
    Ugs = np.loadtxt('out/Ugs.txt') # this is the potential on the tube!
    size_Uds = len(Uds)
    size_Ugs = len(Ugs)

    Ids = np.zeros([size_Uds, size_Ugs])
    Ids = np.loadtxt('out/Ids.txt')

    plt.semilogy(Ugs, -Ids[0,:])
#    plt.show()

########################################################################################################################
    phi_range/const.eV

    phi_range = Ugs*const.eV
    n_phi = len(Ugs)
    cq = np.zeros(n_phi) # quantum capacitance
    for i, phi in enumerate(phi_range):
        my_cq.phi = phi  # change the attribute
        cq[i] = my_cq.compute_integral()

    l = 2*const.nano  # screening length
    my_ci.dummi_ci(l)  # my_ci.empiric_ci is assinged to a new value!

    coef = (my_ci.empiric_ci + cq) / my_ci.empiric_ci  # coef between phi and V VECTOR size: phi_range

    V = coef*phi_range/const.eV
    plt.figure()
    plt.semilogy(-Ugs, Ids[0,:], ':', color = 'C0', label='original')
    plt.semilogy(-Ugs, Ids[1,:], ':', color = 'C1', label='original')
    plt.semilogy(-Ugs, Ids[2,:], ':', color = 'C2', label='original')
    plt.semilogy(-Ugs, Ids[0,:], ':', color = 'C3', label='original')
    plt.semilogy(-V, Ids[0,:], label='Vds = -0.01', color='C0')
    plt.semilogy(-V, Ids[1,:], label='Vds = -0.05', color='C1')
    plt.semilogy(-V, Ids[2,:], label='Vds = -0.1', color='C2')
    plt.semilogy(-V, Ids[3,:], label='Vds = -0.5', color='C3')

    plt.xlim([-2, 2])
    plt.legend()
    plt.show()

    print('I am done')


def get_V(Ugs, uds, V, my_input):
    EF = uds*const.eV
    V = np.zeros(Ugs)
    for i, ugs in range(Ugs):
        V[i] =1


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

    def epsilon_i(self, input):
        #print('2*l/gamma = ', 2*self.l/np.e)
        return 2*np.pi*const.epsilon_0*self.epsilon/np.log(2*self.l/self.R/np.e)

    def dummi_ci(self, l):
        # l is the dipole layer length in nm
        self.l_empiric = l
#        self.empiric_ci = 2*np.pi*self.epsilon*const.epsilon_0/np.log(2*self.l_empiric/self.R/np.e)
        self.empiric_ci = 2*np.pi*self.epsilon*const.epsilon_0/scipy.special.k0(1.0/self.l_empiric*self.R)

#def find_V_from_chi(Cq,Ci,):
#    def F(self,phi):
#        coef = /C2
#        return phi - 1/(1 + coef)*V


def my_sqrt_m1(x):
    if x >= 0:
        return 1.0/np.sqrt(x)
    else:
        return 0

def K0(x):
    return scipy.special.k0(x)

def K0_approx(x):
    return -np.log(x/2)-1/np.e

def plot_phi_vs_Voltage(coef_many_ci, phi_range, l, input):
    plt.figure('phi vs V')
    for i in range(np.size(coef_many_ci,1)):
        plt.plot(coef_many_ci[:,i]*phi_range/const.eV, phi_range/const.eV, label='$l=$ {} nm'.format(l[i]/const.nano))  # voltage(x) vs phi(y)
    plt.ylabel('phi')
    E0 = input['E0']
    E1 = input['E1']
    I = np.ones(len(phi_range))
    EF = input['EF']
    plt.plot(phi_range/const.eV, E0*I+EF, linestyle=':', color = 'grey')
    plt.plot(phi_range/const.eV, E1*I+EF, linestyle=':', color = 'grey')
    plt.plot(phi_range/const.eV, -E0*I+EF, linestyle=':', color = 'grey' )
    plt.plot(phi_range/const.eV, -E1*I+EF, linestyle=':', color = 'grey')
    plt.xlabel('Voltage')
    plt.grid()
#    plt.xlim(-1.5, 1.5)
    plt.legend()
    plt.savefig('phi_vs_V.png')
    #plt.show()
    plt.close()


def plot_l_vs_ci(l, ci):
    plt.figure('l vs V')
    for i in range(len(ci)):
        plt.plot(l/const.nano, ci/const.atto*const.micro)  # voltage(x) vs phi(y)
        plt.semilogy(l / const.nano, ci / const.atto * const.micro)  # voltage(x) vs phi(y)
    plt.xlabel('l, nm')
    plt.ylabel('capacitance, aF/nm')
    plt.grid()
    plt.savefig('ci_vs_l.png')
    #plt.show()
    plt.close()


def test_K0_approx():
    K0(100)
    K0_approx(100)
    n = 1000
    K_exact = np.zeros(n)
    K_approx = np.zeros(n)
    x_range = np.linspace(0.01,10,n)
    for i, x in enumerate(x_range):
        K_exact[i] = K0(x)
        K_approx[i] = K0_approx(x)
    plt.figure('K exact K approx')
    plt.plot(x_range, K_approx, label='approx')
    plt.plot(x_range, K_exact, label='exact', linestyle=':')
    plt.legend()
    plt.close()

if __name__ == '__main__':
    main()
