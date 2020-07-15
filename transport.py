from scipy.integrate import quad
import scipy.constants as const
import scipy
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import yaml
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        help='input file')
    input_file_name = parser.parse_args().input
    with open(input_file_name) as fid:
        my_input = yaml.load(fid, Loader=yaml.SafeLoader)

    # test Fermi
    print(const.k*300/const.eV  )
    #plot_Fermi(np.linspace(-1.5, 1.5, 100), 0.0, my_input)
    # test integration

# test Transmission
    #plot_T(np.linspace(-1.5, 1.5, 100), 0.4, my_input)
    print('done')


# test integrate
    phi_g = 0.0*const.eV
    phi_d_range = np.linspace(-1.0, 1.0, 100)*const.eV
    I = np.zeros(len(phi_d_range))
    for i, phi_d in enumerate(phi_d_range):
        I[i] = quad(fun4int, -3.0 * const.eV, 3 * const.eV, epsabs=1e-17, args=(phi_d, phi_g, my_input))[0]
    plt.plot(phi_d_range/const.eV, I)
    plt.xlabel('phi_d, eV')
    plt.ylabel('current, A')
    plt.close()
    #plt.show()

# test integrate b
    phi_d = 0.5*const.eV
    phi_g_range = np.linspace(-0.5, 0.5, 100)*const.eV
    I = np.zeros(len(phi_d_range))
    for i, phi_g in enumerate(phi_g_range):
        I[i] = quad(fun4int, -4.0 * const.eV, 4 * const.eV, epsabs=1e-20, args=(phi_d, phi_g, my_input))[0]
    plt.semilogy(phi_g_range/const.eV, I)
    plt.xlabel('phi_g, eV')
    plt.ylabel('current, A')
    #plt.show()
    plt.close()


    # test integrate. output charachteristics
    phi_d_range = np.array([0.01, 0.05, 0.1, 0.5 ])*const.eV
    phi_g_range = np.linspace(-1.5, 1.5, 100)*const.eV
    I = np.zeros([len(phi_d_range), len(phi_g_range)])
    for j, phi_d in enumerate(phi_d_range):
        for i, phi_g in enumerate(phi_g_range):
            I[j,i] = quad(fun4int, -2.0 * const.eV, 2 * const.eV, epsabs=1e-18, epsrel=1e-8, args=(phi_d, phi_g, my_input))[0]
    for i, phi_ds in enumerate(phi_d_range):
        plt.semilogy(phi_g_range/const.eV, I[i,:], label = '$\phi_{ds}$ = '+ str(phi_ds/const.eV) + ' eV')
    plt.xlabel('$\phi_g$, eV')
    plt.ylabel('current, A')
    plt.legend()
    plt.show()

def Fermi(energy, phi_d, input):
    # phi: potential [eV]. potential on source: 0
    # doping: position of the Fermi level wrt the center of the band gap. p: minus; n: plus
    #doping = input['doping']*const.eV
    kT = const.k*input['T']
    if kT != 0:
        e = (energy - phi_d)/kT
        return 1.0/(1.0 + np.exp(e))
    else:
        return np.heaviside(energy,0)


def fun4int(energy, phi_d, phi_g, input):
    g0 = const.physical_constants['conductance quantum'][0]
    trans = transmission(energy, phi_g, input)
    drain_flow = trans*Fermi(energy,phi_d, input)
    source_flow = trans*Fermi(energy, 0, input)
    return g0/const.eV*(drain_flow - source_flow)


def transmission(energy, phi_g, my_input):
    E0 = my_input['E11']/2.0*const.eV
    E1 = my_input['E22']/2.0*const.eV
    doping = my_input['doping']*const.eV
    delta = phi_g + doping
    first_band = np.float((np.abs(energy - delta) > E0))
    second_band = np.float((np.abs(energy - delta) > E1))
    return 2*(first_band + second_band) # 2 for spin degeneracy


# tests
def plot_Fermi(energy_range, phi, input):
    doping = input['doping']
    occupation = np.zeros(len(energy_range))
    for i, energy in enumerate(energy_range):
        occupation[i] = Fermi(energy, phi, input)
    plt.plot(energy_range, occupation)
    plt.xlabel('Energy, eV')
    plt.ylabel('Occupation')
    plt.ylim([0.0,1.1])
    if phi > 0:
        plt.legend(['positive potential on drain'])
    if doping > 0:
        plt.legend(['n type doping'])
    if not os.path.isdir('test'):
        os.makedirs('test')
    plt.savefig('test/Fermi.png')
    plt.close()
    #plt.show()


def plot_T(energy_range, phi_g, input):
    t = np.zeros(len(energy_range))
    for i, energy in enumerate(energy_range):
        t[i] = transmission(energy, phi_g, input)
    plt.plot(energy_range, t)
    plt.xlabel('Energy, eV')
    plt.ylabel('Occupation')
    plt.show()
    if not os.path.isdir('test'):
        os.makedirs('test')
    plt.savefig('test/Transmission.png')
    plt.close()




if __name__ == '__main__':
    main()