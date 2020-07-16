import liqued_many_Ci
import transport
from transport import fun4int
from liqued_many_Ci import CQ
from liqued_many_Ci import Ci

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

    # get the input
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        help='input file')
    input_file_name = parser.parse_args().input
    with open(input_file_name) as fid:
        my_input = yaml.load(fid, Loader=yaml.SafeLoader)
    #end: get the input

#################################### BARE TRANSFER CHARACHTERISTICS ####################################################
    # manual input
    phi_d_range = np.array([0.01, 0.05, 0.1, 0.5 ])*const.eV
    phi_g_range = np.linspace(-1.5, 1.5, 300)*const.eV
    n_d = len(phi_d_range)
    n_g = len(phi_g_range)
    I = np.zeros([n_d, n_g])  # current

    for j, phi_d in enumerate(phi_d_range):
        for i, phi_g in enumerate(phi_g_range):
            I[j,i] = quad(fun4int, -3.0 * const.eV, 3 * const.eV, epsabs=1e-17, epsrel=1e-8, limit = 1000, args=(phi_d, phi_g, my_input))[0]
    for i, phi_ds in enumerate(phi_d_range):
        plt.semilogy(phi_g_range/const.eV, I[i,:], label = '$\phi_{ds}$ = '+ str(phi_ds/const.eV) + ' eV')
    plt.xlabel('$\phi_g$, eV')
    plt.ylabel('current, A')
    plt.legend()
    plt.show()


############################################ CAPACITY ##################################################################
    # manual input
    l = 1.5 * const.nano  # screening length


    my_ci = Ci(my_input)
    my_ci.dummi_ci(l)  # my_ci.empiric_ci is assinged to a new value!

    my_cq = CQ(my_input)
    cq = np.zeros(n_g) # quantum capacitance (source)
    for i, phi_g in enumerate(phi_g_range): # ????
        my_cq.integrand(phi_g)
        cq[i] = my_cq.compute_integral(phi_g)
    cq_d = np.zeros([n_d, n_g]) # the same (drain)
    for i, phi_g in enumerate(phi_g_range): #drain
        for j, phi_d in enumerate(phi_d_range):
            my_cq.integrand(phi_g+phi_d)
            cq_d[j,i] = my_cq.compute_integral(phi_g+phi_d)

    coef = (my_ci.empiric_ci + cq) / my_ci.empiric_ci   # coef between phi and V VECTOR size: phi_range
    coef_d = (my_ci.empiric_ci + cq_d) / my_ci.empiric_ci   # coef between phi and V VECTOR size: phi_range
    coef_average = 0.5*(coef+coef_d)

    phi_g_appl_range = coef*phi_g_range
    phi_g_appl_range_d = coef_d*phi_g_range

    plt.plot(phi_g_range/const.eV, phi_g_appl_range/const.eV, label='source')
    plt.plot(phi_g_range/const.eV, phi_g_appl_range_d[-1,:]/const.eV, label='drain 0.5')
    plt.plot(phi_g_range/const.eV, 0.5*(phi_g_appl_range_d[-1,:]+phi_g_appl_range)/const.eV, label='average')
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()
################################################ SCREENED TRANSFER CHARACHTERISTICS ####################################

    for i in range(n_d):
        plt.semilogy(-phi_g_range/const.eV, I[i,:], linestyle = ':', label = '$\phi_{ds}$ = '+ str(phi_ds/const.eV) + ' eV', color = 'C'+str(i))
        plt.semilogy(fid-phi_g_appl_range/const.eV, I[i,:],color = 'C'+str(i))
#        plt.semilogy(-phi_g_appl_range_d[i,:]/const.eV, I[i,:],color = 'C'+str(i), marker = 'x')
#        plt.semilogy(-0.5 * (phi_g_appl_range_d[i, :] + phi_g_appl_range) / const.eV, I[i, :], color='C' + str(i),
#                 marker='o')
    plt.xlabel('$V_{gs}$, V')
    plt.grid()
    plt.xlim([-1.5, 1.5])
    plt.show()



if __name__ == '__main__':
    main()