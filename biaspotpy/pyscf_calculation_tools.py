import pyscf
import glob
import os

import numpy as np

from pyscf.hessian import thermo

from calc_tools import Calculationtools
from optimizer import Model_hess_tmp
from param import UnitValueLib

class SinglePoint:
    def __init__(self, **kwarg):
        UVL = UnitValueLib()

        self.bohr2angstroms = UVL.bohr2angstroms
        
        self.START_FILE = kwarg["START_FILE"]
        self.SUB_BASIS_SET = kwarg["SUB_BASIS_SET"]
        self.BASIS_SET = kwarg["BASIS_SET"]
        self.N_THREAD = kwarg["N_THREAD"]
        self.SET_MEMORY = kwarg["SET_MEMORY"]
        self.FUNCTIONAL = kwarg["FUNCTIONAL"]
        self.FC_COUNT = kwarg["FC_COUNT"]
        self.BPA_FOLDER_DIRECTORY = kwarg["BPA_FOLDER_DIRECTORY"]
        self.Model_hess = kwarg["Model_hess"]
    
    
    def pyscf_calculation(self, file_directory, element_list, iter):
        """execute QM calclation."""
        gradient_list = []
        energy_list = []
        geometry_num_list = []
        geometry_optimized_num_list = []
        finish_frag = False
        try:
            os.mkdir(file_directory)
        except:
            pass
        file_list = glob.glob(file_directory+"/*_[0-9].xyz")
        for num, input_file in enumerate(file_list):
            try:
                
                pyscf.lib.num_threads(self.N_THREAD)
                
                with open(input_file, "r") as f:
                    words = f.readlines()
                input_data_for_display = []
                for word in words[2:]:
                    input_data_for_display.append(np.array(word.split()[1:4], dtype="float64")/self.bohr2angstroms)
                input_data_for_display = np.array(input_data_for_display, dtype="float64")
                
                print("\n",input_file,"\n")
                mol = pyscf.gto.M(atom = input_file,
                                  charge = self.electronic_charge,
                                  spin = self.spin_multiplicity,
                                  basis = self.SUB_BASIS_SET,
                                  max_memory = float(self.SET_MEMORY.replace("GB","")) * 1024, #SET_MEMORY unit is GB
                                  verbose=3)
                if self.FUNCTIONAL == "hf" or self.FUNCTIONAL == "HF":
                    if int(self.spin_multiplicity) > 0:
                        mf = mol.UHF().x2c().density_fit()
                    else:
                        mf = mol.RHF().density_fit()
                else:
                    if int(self.spin_multiplicity) > 1:
                        mf = mol.UKS().x2c().density_fit()
                    else:
                        mf = mol.RKS().density_fit()
                    mf.xc = self.FUNCTIONAL
   
            
          
                g = mf.run().nuc_grad_method().kernel()
                e = float(vars(mf)["e_tot"])
                g = np.array(g, dtype = "float64")

                print("\n")


                if self.FC_COUNT == -1:
                    pass
                
                elif iter % self.FC_COUNT == 0:
                    
                    """exact hessian"""
                    exact_hess = mf.Hessian().kernel()
                    
                    freqs = thermo.harmonic_analysis(mf.mol, exact_hess)
                    exact_hess = exact_hess.transpose(0,2,1,3).reshape(len(input_data_for_display)*3, len(input_data_for_display)*3)
                    print("frequencies: \n",freqs["freq_wavenumber"])
                    eigenvalues, _ = np.linalg.eigh(exact_hess)
                    print("=== hessian (before add bias potential) ===")
                    print("eigenvalues: ", eigenvalues)
                    exact_hess = Calculationtools().project_out_hess_tr_and_rot(exact_hess, element_list, input_data_for_display)

                    self.Model_hess = Model_hess_tmp(exact_hess, momentum_disp=self.Model_hess.momentum_disp, momentum_grad=self.Model_hess.momentum_grad)

            except Exception as error:
                print(error)
                print("This molecule could not be optimized.")
                finish_frag = True
                return 0, 0, 0, finish_frag   
            
      
        return e, g, input_data_for_display, finish_frag