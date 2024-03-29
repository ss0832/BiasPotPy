import argparse
import sys
import os
import copy
import glob
import itertools
import datetime
import time

import numpy as np

from optimizer import CalculateMoveVector, Opt_calc_tmps, Model_hess_tmp
from potential import BiasPotentialCalculation
from calc_tools import CalculationStructInfo, Calculationtools
from visualization import Graph
from fileio import FileIO
from parameter import UnitValueLib, element_number

try:
    import psi4
    
except:
    pass

"""
    BiasPotPy
    Copyright (C) 2023-2024 ss0832

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
#please input psi4 inputfile.
XMOL format (Enter the formal charge and spin multiplicity on the comment line, e.g., "0 1")
....
"""

"""
references:

Psi4
 D. G. A. Smith, L. A. Burns, A. C. Simmonett, R. M. Parrish, M. C. Schieber, R. Galvelis, P. Kraus, H. Kruse, R. Di Remigio, A. Alenaizan, A. M. James, S. Lehtola, J. P. Misiewicz, M. Scheurer, R. A. Shaw, J. B. Schriber, Y. Xie, Z. L. Glick, D. A. Sirianni, J. S. O'Brien, J. M. Waldrop, A. Kumar, E. G. Hohenstein, B. P. Pritchard, B. R. Brooks, H. F. Schaefer III, A. Yu. Sokolov, K. Patkowski, A. E. DePrince III, U. Bozkaya, R. A. King, F. A. Evangelista, J. M. Turney, T. D. Crawford, C. D. Sherrill, "Psi4 1.4: Open-Source Software for High-Throughput Quantum Chemistry", J. Chem. Phys. 152(18) 184108 (2020).
 
PySCF
Recent developments in the PySCF program package, Qiming Sun, Xing Zhang, Samragni Banerjee, Peng Bao, Marc Barbry, Nick S. Blunt, Nikolay A. Bogdanov, George H. Booth, Jia Chen, Zhi-Hao Cui, Janus J. Eriksen, Yang Gao, Sheng Guo, Jan Hermann, Matthew R. Hermes, Kevin Koh, Peter Koval, Susi Lehtola, Zhendong Li, Junzi Liu, Narbe Mardirossian, James D. McClain, Mario Motta, Bastien Mussard, Hung Q. Pham, Artem Pulkin, Wirawan Purwanto, Paul J. Robinson, Enrico Ronca, Elvira R. Sayfutyarova, Maximilian Scheurer, Henry F. Schurkus, James E. T. Smith, Chong Sun, Shi-Ning Sun, Shiv Upadhyay, Lucas K. Wagner, Xiao Wang, Alec White, James Daniel Whitfield, Mark J. Williamson, Sebastian Wouters, Jun Yang, Jason M. Yu, Tianyu Zhu, Timothy C. Berkelbach, Sandeep Sharma, Alexander Yu. Sokolov, and Garnet Kin-Lic Chan, J. Chem. Phys., 153, 024109 (2020). doi:10.1063/5.0006074

GFN2-xTB(tblite)
J. Chem. Theory Comput. 2019, 15, 3, 1652–1671 
GFN-xTB(tblite)
J. Chem. Theory Comput. 2017, 13, 5, 1989–2009
"""

def ieipparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", help='input folder')
    parser.add_argument("-bs", "--basisset", default='6-31G(d)', help='basisset (ex. 6-31G*)')
    parser.add_argument("-func", "--functional", default='b3lyp', help='functional(ex. b3lyp)')
    parser.add_argument("-sub_bs", "--sub_basisset", type=str, nargs="*", default='', help='sub_basisset (ex. I LanL2DZ)')
    parser.add_argument("-gfix", "--gradient_fix_atoms", nargs="*",  type=str, default="", help='set the gradient of internal coordinates between atoms to zero  (ex.) [[atoms (ex.) 1,2] ...]')
    parser.add_argument("-core", "--N_THREAD",  type=int, default='8', help='threads')
    parser.add_argument("-mi", "--microiter",  type=int, default=0, help='microiteration for relaxing reaction pathways')
    parser.add_argument("-beta", "--BETA",  type=float, default='1.0', help='force for optimization')
    parser.add_argument("-mem", "--SET_MEMORY",  type=str, default='2GB', help='use mem(ex. 1GB)')

    parser = parser_for_biasforce(parser)
    
    parser.add_argument("-xtb", "--usextb",  type=str, default="None", help='use extended tight bonding method to calculate. default is not using extended tight binding method (ex.) GFN1-xTB, GFN2-xTB ')
    parser.add_argument('-pyscf','--pyscf', help="use pyscf module.", action='store_true')
    parser.add_argument('-u','--unrestrict', help="use unrestricted method (for radical reaction and excite state etc.)", action='store_true')
    parser.add_argument("-elec", "--electronic_charge", type=int, default=0, help='formal electronic charge (ex.) [charge (0)]')
    parser.add_argument("-spin", "--spin_multiplicity", type=int, default=1, help='spin multiplcity (if you use pyscf, please input S value (mol.spin = 2S = Nalpha - Nbeta)) (ex.) [multiplcity (0)]')
    
    
    args = parser.parse_args()
    args.fix_atoms = []
    args.gradient_fix_atoms = []
    args.geom_info = ["0"]
    args.opt_method = ""    
    args.opt_fragment = []
    return args




def optimizeparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", help='input xyz file name')
    parser.add_argument("-bs", "--basisset", default='6-31G(d)', help='basisset (ex. 6-31G*)')
    parser.add_argument("-func", "--functional", default='b3lyp', help='functional(ex. b3lyp)')
    parser.add_argument("-sub_bs", "--sub_basisset", type=str, nargs="*", default='', help='sub_basisset (ex. I LanL2DZ)')

    parser.add_argument("-ns", "--NSTEP",  type=int, default='300', help='iter. number')
    parser.add_argument("-core", "--N_THREAD",  type=int, default='8', help='threads')
    parser.add_argument("-mem", "--SET_MEMORY",  type=str, default='2GB', help='use mem(ex. 1GB)')
    parser.add_argument("-d", "--DELTA",  type=str, default='x', help='move step')
    parser.add_argument('-u','--unrestrict', help="use unrestricted method (for radical reaction and excite state etc.)", action='store_true')
    parser = parser_for_biasforce(parser)
    
    parser.add_argument("-fix", "--fix_atoms", nargs="*",  type=str, default="", help='fix atoms (ex.) [atoms (ex.) 1,2,3-6]')
    parser.add_argument("-gfix", "--gradient_fix_atoms", nargs="*",  type=str, default="", help='set the gradient of internal coordinates between atoms to zero  (ex.) [[atoms (ex.) 1,2] ...]')
    parser.add_argument("-md", "--md_like_perturbation",  type=str, default="0.0", help='add perturbation like molecule dynamics (ex.) [[temperature (unit. K)]]')
    parser.add_argument("-gi", "--geom_info", nargs="*",  type=str, default="1", help='calculate atom distances, angles, and dihedral angles in every iteration (energy_profile is also saved.) (ex.) [atoms (ex.) 1,2,3-6]')
    parser.add_argument("-opt", "--opt_method", nargs="*", type=str, default=["AdaBelief"], help='optimization method for QM calclation (default: AdaBelief) (mehod_list:(steepest descent method group) RADAM, AdaBelief, AdaDiff, EVE, AdamW, Adam, Adadelta, Adafactor, Prodigy, NAdam, AdaMax, FIRE, conjugate_gradient_descent (quasi-Newton method group) mBFGS, mFSB, RFO_mBFGS, RFO_mFSB, FSB, RFO_FSB, BFGS, RFO_BFGS, TRM_FSB, TRM_BFGS) (notice you can combine two methods, steepest descent family and quasi-Newton method family. The later method is used if gradient is small enough. [[steepest descent] [quasi-Newton method]]) (ex.) [opt_method]')
    parser.add_argument("-fc", "--calc_exact_hess",  type=int, default=-1, help='calculate exact hessian per steps (ex.) [steps per one hess calculation]')
    parser.add_argument("-xtb", "--usextb",  type=str, default="None", help='use extended tight bonding method to calculate. default is not using extended tight binding method (ex.) GFN1-xTB, GFN2-xTB ')
    parser.add_argument('-pyscf','--pyscf', help="use pyscf module.", action='store_true')
    parser.add_argument("-elec", "--electronic_charge", type=int, default=0, help='formal electronic charge (ex.) [charge (0)]')
    parser.add_argument("-spin", "--spin_multiplicity", type=int, default=1, help='spin multiplcity (if you use pyscf, please input S value (mol.spin = 2S = Nalpha - Nbeta)) (ex.) [multiplcity (0)]')
    parser.add_argument("-order", "--saddle_order", type=int, default=0, help='optimization for (n-1)-th order saddle point (Newton group of opt method (RFO) is only available.) (ex.) [order (0)]')
    parser.add_argument('-cmds','--cmds', help="apply classical multidimensional scaling to calculated approx. reaction path.", action='store_true')
    parser.add_argument('-rc','--ricci_curvature', help="calculate Ricci scalar of calculated approx. reaction path.", action='store_true')
    parser.add_argument("-of", "--opt_fragment", nargs="*", type=str, default=[], help="Several atoms are grouped together as fragments and optimized. (ex.) [[atoms (ex.) 1-4] ...] ")#(2024/3/26) this option doesn't work if you use quasi-Newton method for optimization.
    
    args = parser.parse_args()
    return args

def parser_for_biasforce(parser):
    parser.add_argument("-ma", "--manual_AFIR", nargs="*",  type=str, default=['0.0', '1', '2'], help='manual-AFIR (ex.) [[Gamma(kJ/mol)] [Fragm.1(ex. 1,2,3-5)] [Fragm.2] ...]')
    parser.add_argument("-rp", "--repulsive_potential", nargs="*",  type=str, default=['0.0','1.0', '1', '2', 'scale'], help='Add LJ repulsive_potential based on UFF (ex.) [[well_scale] [dist_scale] [Fragm.1(ex. 1,2,3-5)] [Fragm.2] [scale or value(kJ/mol ang.)] ...]')
    parser.add_argument("-rpv2", "--repulsive_potential_v2", nargs="*",  type=str, default=['0.0','1.0','0.0','1','2','12','6', '1,2', '1-2', 'scale'], help='Add LJ repulsive_potential based on UFF (ver.2) (eq. V = ε[A * (σ/r)^(rep) - B * (σ/r)^(attr)]) (ex.) [[well_scale] [dist_scale] [length (ang.)] [const. (rep)] [const. (attr)] [order (rep)] [order (attr)] [LJ center atom (1,2)] [target atoms (3-5,8)] [scale or value(kJ/mol ang.)] ...]')
    parser.add_argument("-rpg", "--repulsive_potential_gaussian", nargs="*",  type=str, default=['0.0','1.0','0.0','1.0','1.0', '1,2', '1-2'], help='Add LJ repulsive_potential based on UFF (ver.2) (eq. V = ε_LJ[(σ/r)^(12) - 2 * (σ/r)^(6)] - ε_gau * exp(-((r-σ_gau)/b)^2)) (ex.) [[LJ_well_depth (kJ/mol)] [LJ_dist (ang.)] [Gaussian_well_depth (kJ/mol)] [Gaussian_dist (ang.)] [Gaussian_range (ang.)] [Fragm.1 (1,2)] [Fragm.2 (3-5,8)] ...]')
    
    
    parser.add_argument("-cp", "--cone_potential", nargs="*",  type=str, default=['0.0','1.0','90','1', '2,3,4', '5-9'], help='Add cone type LJ repulsive_potential based on UFF (ex.) [[well_value (epsilon) (kJ/mol)] [dist (sigma) (ang.)] [cone angle (deg.)] [LJ center atom (1)] [three atoms (2,3,4) ] [target atoms (5-9)] ...]')
    
    
    parser.add_argument("-kp", "--keep_pot", nargs="*",  type=str, default=['0.0', '1.0', '1,2'], help='keep potential 0.5*k*(r - r0)^2 (ex.) [[spring const.(a.u.)] [keep distance (ang.)] [atom1,atom2] ...] ')
    parser.add_argument("-kpv2", "--keep_pot_v2", nargs="*",  type=str, default=['0.0', '1.0', '1', '2'], help='keep potential_v2 0.5*k*(r - r0)^2 (ex.) [[spring const.(a.u.)] [keep distance (ang.)] [Fragm.1] [Fragm.2] ...] ')
    parser.add_argument("-akp", "--anharmonic_keep_pot", nargs="*",  type=str, default=['0.0', '1.0', '1.0', '1,2'], help='Morse potential  De*[1-exp(-((k/2*De)^0.5)*(r - r0))]^2 (ex.) [[potential well depth (a.u.)] [spring const.(a.u.)] [keep distance (ang.)] [atom1,atom2] ...] ')
    parser.add_argument("-ka", "--keep_angle", nargs="*",  type=str, default=['0.0', '90', '1,2,3'], help='keep angle 0.5*k*(θ - θ0)^2 (0 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep angle (degrees)] [atom1,atom2,atom3] ...] ')
    parser.add_argument("-kav2", "--keep_angle_v2", nargs="*",  type=str, default=['0.0', '90', '1', '2', '3'], help='keep angle_v2 0.5*k*(θ - θ0)^2 (0 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep angle (degrees)] [Fragm.1] [Fragm.2] [Fragm.3] ...] ')
    
    parser.add_argument("-ddka", "--atom_distance_dependent_keep_angle", nargs="*",  type=str, default=['0.0', '90', "120", "1.4", "5", "1", '2,3,4'], help='atom-distance-dependent keep angle (ex.) [[spring const.(a.u.)] [minimum keep angle (degrees)] [maximum keep angle (degrees)] [base distance (ang.)] [reference atom (1 atom)] [center atom (1 atom)] [atom1,atom2,atom3] ...] ')
    
    
    parser.add_argument("-kda", "--keep_dihedral_angle", nargs="*",  type=str, default=['0.0', '90', '1,2,3,4'], help='keep dihedral angle 0.5*k*(φ - φ0)^2 (-180 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep dihedral angle (degrees)] [atom1,atom2,atom3,atom4] ...] ')
    parser.add_argument("-kdav2", "--keep_dihedral_angle_v2", nargs="*",  type=str, default=['0.0', '90', '1','2','3','4'], help='keep dihedral angle_v2 0.5*k*(φ - φ0)^2 (-180 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep dihedral angle (degrees)] [Fragm.1] [Fragm.2] [Fragm.3] [Fragm.4] ...] ')
    parser.add_argument("-vpp", "--void_point_pot", nargs="*",  type=str, default=['0.0', '1.0', '0.0,0.0,0.0', '1',"2.0"], help='void point keep potential (ex.) [[spring const.(a.u.)] [keep distance (ang.)] [void_point (x,y,z) (ang.)] [atoms(ex. 1,2,3-5)] [order p "(1/p)*k*(r - r0)^p"] ...] ')
   
    parser.add_argument("-wp", "--well_pot", nargs="*", type=str, default=['0.0','1','2','0.5,0.6,1.5,1.6'], help="Add potential to limit atom distance. (ex.) [[wall energy (kJ/mol)] [fragm.1] [fragm.2] [a,b,c,d (a<b<c<d) (ang.)] ...]")
    parser.add_argument("-wwp", "--wall_well_pot", nargs="*", type=str, default=['0.0','x','0.5,0.6,1.5,1.6',"1"], help="Add potential to limit atoms movement. (like sandwich) (ex.) [[wall energy (kJ/mol)] [direction (x,y,z)] [a,b,c,d (a<b<c<d) (ang.)]  [target atoms (1,2,3-5)] ...]")
    parser.add_argument("-vpwp", "--void_point_well_pot", nargs="*", type=str, default=['0.0','0.0,0.0,0.0','0.5,0.6,1.5,1.6',"1"], help="Add potential to limit atom movement. (like sphere) (ex.) [[wall energy (kJ/mol)] [coordinate (x,y,z) (ang.)] [a,b,c,d (a<b<c<d) (ang.)]  [target atoms (1,2,3-5)] ...]")
    parser.add_argument("-awp", "--around_well_pot", nargs="*", type=str, default=['0.0','1','0.5,0.6,1.5,1.6',"2"], help="Add potential to limit atom movement. (like sphere around fragment) (ex.) [[wall energy (kJ/mol)] [center (1,2-4)] [a,b,c,d (a<b<c<d) (ang.)]  [target atoms (2,3-5)] ...]")
    return parser


def nebparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", help='input folder')
    parser.add_argument("-bs", "--basisset", default='6-31G(d)', help='basisset (ex. 6-31G*)')
    parser.add_argument("-func", "--functional", default='b3lyp', help='functional(ex. b3lyp)')
    parser.add_argument('-u','--unrestrict', help="use unrestricted method (for radical reaction and excite state etc.)", action='store_true')
    parser.add_argument("-ns", "--NSTEP",  type=int, default='10', help='iter. number')
    parser.add_argument("-om", "--OM", action='store_true', help='J. Chem. Phys. 155, 074103 (2021)  doi:https://doi.org/10.1063/5.0059593 This improved NEB method is inspired by the Onsager-Machlup (OM) action.')
    parser.add_argument("-lup", "--LUP", action='store_true', help='J. Chem. Phys. 92, 1510–1511 (1990) doi:https://doi.org/10.1063/1.458112 locally updated planes (LUP) method')
    parser.add_argument("-dneb", "--DNEB", action='store_true', help='J. Chem. Phys. 120, 2082–2094 (2004) doi:https://doi.org/10.1063/1.1636455 doubly NEB method (DNEB) method')
    parser.add_argument("-nesb", "--NESB", action='store_true', help='J Comput Chem. 2023;44:1884–1897. https://doi.org/10.1002/jcc.27169 Nudged elastic stiffness band (NESB) method')
    parser.add_argument("-p", "--partition",  type=int, default='0', help='number of nodes')
    parser.add_argument("-core", "--N_THREAD",  type=int, default='8', help='threads')
    parser.add_argument("-mem", "--SET_MEMORY",  type=str, default='1GB', help='use mem(ex. 1GB)')
    parser.add_argument("-cineb", "--apply_CI_NEB",  type=int, default='99999', help='apply CI_NEB method')
    parser.add_argument("-sd", "--steepest_descent",  type=int, default='99999', help='apply steepest_descent method')
    parser.add_argument("-qnt", "--QUASI_NEWTOM_METHOD", action='store_true', help='changing default optimizer to quasi-Newton method')
    parser.add_argument("-xtb", "--usextb",  type=str, default="None", help='use extended tight bonding method to calculate. default is not using extended tight binding method (ex.) GFN1-xTB, GFN2-xTB ')
    parser.add_argument("-fe", "--fixedges",  type=int, default=0, help='fix edges of nodes (1=initial_node, 2=end_node, 3=both_nodes) ')
    parser.add_argument("-aneb", "--ANEB_num",  type=int, default=0, help='execute adaptic NEB (ANEB) method. (default setting is not executing ANEB.)')
    parser.add_argument("-gfix", "--gradient_fix_atoms", nargs="*",  type=str, default="", help='set the gradient of internal coordinates between atoms to zero  (ex.) [[atoms (ex.) 1,2] ...]')
    parser = parser_for_biasforce(parser)
    args = parser.parse_args()
    args.fix_atoms = []
    args.gradient_fix_atoms = []
    args.geom_info = ["0"]
    args.opt_method = ""
    args.opt_fragment = []
    return args


def mdparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", help='input psi4 files')
    parser.add_argument("-bs", "--basisset", default='6-31G(d)', help='basisset (ex. 6-31G*)')
    parser.add_argument("-func", "--functional", default='b3lyp', help='functional(ex. b3lyp)')
    parser.add_argument("-sub_bs", "--sub_basisset", type=str, nargs="*", default='', help='sub_basisset (ex. I LanL2DZ)')

    parser.add_argument("-time", "--NSTEP",  type=int, default='100000', help='time scale')
    parser.add_argument("-traj", "--TRAJECTORY",  type=int, default='10', help='number of trajectory to generate (default) 10')
   
    parser.add_argument("-temp", "--temperature",  type=float, default='298.15', help='temperature [unit. K] (default) 298.15 K')
    parser.add_argument("-press", "--pressure",  type=float, default='1013', help='pressure [unit. kPa] (default) 1013 kPa')
    
    parser.add_argument("-core", "--N_THREAD",  type=int, default='8', help='threads')
    parser.add_argument("-mem", "--SET_MEMORY",  type=str, default='1GB', help='use mem(ex. 1GB)')
    parser.add_argument('-u','--unrestrict', help="use unrestricted method (for radical reaction and excite state etc.)", action='store_true')
    parser.add_argument("-mt", "--mdtype",  type=str, default='nosehoover', help='specify condition to do MD (ex.) velocityverlet (default) nosehoover')
    
    parser.add_argument("-fix", "--fix_atoms", nargs="*",  type=str, default="", help='fix atoms (ex.) [atoms (ex.) 1,2,3-6]')
    parser.add_argument("-gi", "--geom_info", nargs="*",  type=str, default="1", help='calculate atom distances, angles, and dihedral angles in every iteration (energy_profile is also saved.) (ex.) [atoms (ex.) 1,2,3-6]')
    parser.add_argument('-pyscf','--pyscf', help="use pyscf module.", action='store_true')
    parser.add_argument("-elec", "--electronic_charge", type=int, default=0, help='formal electronic charge (ex.) [charge (0)]')
    parser.add_argument("-spin", "--spin_multiplicity", type=int, default=1, help='spin multiplcity (if you use pyscf, please input S value (mol.spin = 2S = Nalpha - Nbeta)) (ex.) [multiplcity (0)]')
    parser.add_argument("-order", "--saddle_order", type=int, default=0, help='optimization for (n-1)-th order saddle point (Newton group of opt method (RFO) is only available.) (ex.) [order (0)]')
    parser.add_argument('-cmds','--cmds', help="apply classical multidimensional scaling to calculated approx. reaction path.", action='store_true')
    parser.add_argument("-xtb", "--usextb",  type=str, default="GFN2-xTB", help='use extended tight bonding method to calculate. default is GFN2-xTB (ex.) GFN1-xTB, GFN2-xTB ')
    
    parser = parser_for_biasforce(parser)
    args = parser.parse_args()
    args.geom_info = ["0"]
    args.gradient_fix_atoms = []
    args.opt_method = ""
    args.opt_fragment = []
    return args

       
def force_data_parser(args):
    def num_parse(numbers):
        sub_list = []
        
        sub_tmp_list = numbers.split(",")
        for sub in sub_tmp_list:                        
            if "-" in sub:
                for j in range(int(sub.split("-")[0]),int(sub.split("-")[1])+1):
                    sub_list.append(j)
            else:
                sub_list.append(int(sub))    
        return sub_list
    force_data = {}
    #---------------------
    if len(args.repulsive_potential) % 5 != 0:
        print("invaild input (-rp)")
        sys.exit(0)
    
    force_data["repulsive_potential_well_scale"] = []
    force_data["repulsive_potential_dist_scale"] = []
    force_data["repulsive_potential_Fragm_1"] = []
    force_data["repulsive_potential_Fragm_2"] = []
    force_data["repulsive_potential_unit"] = []
    
    for i in range(int(len(args.repulsive_potential)/5)):
        force_data["repulsive_potential_well_scale"].append(float(args.repulsive_potential[5*i]))
        force_data["repulsive_potential_dist_scale"].append(float(args.repulsive_potential[5*i+1]))
        force_data["repulsive_potential_Fragm_1"].append(num_parse(args.repulsive_potential[5*i+2]))
        force_data["repulsive_potential_Fragm_2"].append(num_parse(args.repulsive_potential[5*i+3]))
        force_data["repulsive_potential_unit"].append(str(args.repulsive_potential[5*i+4]))
    

    #---------------------
    if len(args.repulsive_potential_v2) % 10 != 0:
        print("invaild input (-rpv2)")
        sys.exit(0)
    
    force_data["repulsive_potential_v2_well_scale"] = []
    force_data["repulsive_potential_v2_dist_scale"] = []
    force_data["repulsive_potential_v2_length"] = []
    force_data["repulsive_potential_v2_const_rep"] = []
    force_data["repulsive_potential_v2_const_attr"] = []
    force_data["repulsive_potential_v2_order_rep"] = []
    force_data["repulsive_potential_v2_order_attr"] = []
    force_data["repulsive_potential_v2_center"] = []
    force_data["repulsive_potential_v2_target"] = []
    force_data["repulsive_potential_v2_unit"] = []
    
    for i in range(int(len(args.repulsive_potential_v2)/10)):
        force_data["repulsive_potential_v2_well_scale"].append(float(args.repulsive_potential_v2[10*i+0]))
        force_data["repulsive_potential_v2_dist_scale"].append(float(args.repulsive_potential_v2[10*i+1]))
        force_data["repulsive_potential_v2_length"].append(float(args.repulsive_potential_v2[10*i+2]))
        force_data["repulsive_potential_v2_const_rep"].append(float(args.repulsive_potential_v2[10*i+3]))
        force_data["repulsive_potential_v2_const_attr"].append(float(args.repulsive_potential_v2[10*i+4]))
        force_data["repulsive_potential_v2_order_rep"].append(float(args.repulsive_potential_v2[10*i+5]))
        force_data["repulsive_potential_v2_order_attr"].append(float(args.repulsive_potential_v2[10*i+6]))
        force_data["repulsive_potential_v2_center"].append(num_parse(args.repulsive_potential_v2[10*i+7]))
        force_data["repulsive_potential_v2_target"].append(num_parse(args.repulsive_potential_v2[10*i+8]))
        force_data["repulsive_potential_v2_unit"].append(str(args.repulsive_potential_v2[10*i+9]))
        if len(force_data["repulsive_potential_v2_center"][i]) != 2:
            print("invaild input (-rpv2 center)")
            sys.exit(0)

    #---------------------
    if len(args.repulsive_potential_gaussian) % 7 != 0:
        print("invaild input (-rpg)")
        sys.exit(0)

    force_data["repulsive_potential_gaussian_LJ_well_depth"] = []
    force_data["repulsive_potential_gaussian_LJ_dist"] = []
    force_data["repulsive_potential_gaussian_gau_well_depth"] = []
    force_data["repulsive_potential_gaussian_gau_dist"] = []
    force_data["repulsive_potential_gaussian_gau_range"] = []
    force_data["repulsive_potential_gaussian_fragm_1"] = []
    force_data["repulsive_potential_gaussian_fragm_2"] = []

    
    for i in range(int(len(args.repulsive_potential_gaussian)/7)):
        force_data["repulsive_potential_gaussian_LJ_well_depth"].append(float(args.repulsive_potential_gaussian[7*i+0]))
        force_data["repulsive_potential_gaussian_LJ_dist"].append(float(args.repulsive_potential_gaussian[7*i+1]))
        force_data["repulsive_potential_gaussian_gau_well_depth"].append(float(args.repulsive_potential_gaussian[7*i+2]))
        force_data["repulsive_potential_gaussian_gau_dist"].append(float(args.repulsive_potential_gaussian[7*i+3]))
        force_data["repulsive_potential_gaussian_gau_range"].append(float(args.repulsive_potential_gaussian[7*i+4]))
        force_data["repulsive_potential_gaussian_fragm_1"].append(num_parse(args.repulsive_potential_gaussian[7*i+5]))
        force_data["repulsive_potential_gaussian_fragm_2"].append(num_parse(args.repulsive_potential_gaussian[7*i+6]))
       

    #---------------------    
    if len(args.cone_potential) % 6 != 0:
        print("invaild input (-cp)")
        sys.exit(0)
    
    force_data["cone_potential_well_value"] = []
    force_data["cone_potential_dist_value"] = []
    force_data["cone_potential_cone_angle"] = []
    force_data["cone_potential_center"] = []
    force_data["cone_potential_three_atoms"] = []
    force_data["cone_potential_target"] = []

    for i in range(int(len(args.cone_potential)/6)):
        force_data["cone_potential_well_value"].append(float(args.cone_potential[6*i+0]))
        force_data["cone_potential_dist_value"].append(float(args.cone_potential[6*i+1]))
        force_data["cone_potential_cone_angle"].append(float(args.cone_potential[6*i+2]))
        force_data["cone_potential_center"].append(int(args.cone_potential[6*i+3]))
        force_data["cone_potential_three_atoms"].append(num_parse(args.cone_potential[6*i+4]))
        force_data["cone_potential_target"].append(num_parse(args.cone_potential[6*i+5]))

        if len(force_data["cone_potential_three_atoms"][i]) != 3:
            print("invaild input (-cp three atoms)")
            sys.exit(0)               
                            
    
    #--------------------
    if len(args.manual_AFIR) % 3 != 0:
        print("invaild input (-ma)")
        sys.exit(0)
    
    force_data["AFIR_gamma"] = []
    force_data["AFIR_Fragm_1"] = []
    force_data["AFIR_Fragm_2"] = []
    

    for i in range(int(len(args.manual_AFIR)/3)):
        force_data["AFIR_gamma"].append(float(args.manual_AFIR[3*i]))#kj/mol
        force_data["AFIR_Fragm_1"].append(num_parse(args.manual_AFIR[3*i+1]))
        force_data["AFIR_Fragm_2"].append(num_parse(args.manual_AFIR[3*i+2]))
    
    
    #---------------------
    if len(args.anharmonic_keep_pot) % 4 != 0:
        print("invaild input (-akp)")
        sys.exit(0)
    
    force_data["anharmonic_keep_pot_potential_well_depth"] = []
    force_data["anharmonic_keep_pot_spring_const"] = []
    force_data["anharmonic_keep_pot_distance"] = []
    force_data["anharmonic_keep_pot_atom_pairs"] = []
    
    for i in range(int(len(args.anharmonic_keep_pot)/4)):
        force_data["anharmonic_keep_pot_potential_well_depth"].append(float(args.anharmonic_keep_pot[4*i]))#au
        force_data["anharmonic_keep_pot_spring_const"].append(float(args.anharmonic_keep_pot[4*i+1]))#au
        force_data["anharmonic_keep_pot_distance"].append(float(args.anharmonic_keep_pot[4*i+2]))#ang
        force_data["anharmonic_keep_pot_atom_pairs"].append(num_parse(args.anharmonic_keep_pot[4*i+3]))
        if len(force_data["anharmonic_keep_pot_atom_pairs"][i]) != 2:
            print("invaild input (-akp atom_pairs)")
            sys.exit(0)
        
    #---------------------
    if len(args.keep_pot) % 3 != 0:
        print("invaild input (-kp)")
        sys.exit(0)
    
    force_data["keep_pot_spring_const"] = []
    force_data["keep_pot_distance"] = []
    force_data["keep_pot_atom_pairs"] = []
    
    for i in range(int(len(args.keep_pot)/3)):
        force_data["keep_pot_spring_const"].append(float(args.keep_pot[3*i]))#au
        force_data["keep_pot_distance"].append(float(args.keep_pot[3*i+1]))#ang
        force_data["keep_pot_atom_pairs"].append(num_parse(args.keep_pot[3*i+2]))
        if len(force_data["keep_pot_atom_pairs"][i]) != 2:
            print("invaild input (-kp atom_pairs)")
            sys.exit(0)
        
    #---------------------
    if len(args.keep_pot_v2) % 4 != 0:
        print("invaild input (-kpv2)")
        sys.exit(0)
    
    force_data["keep_pot_v2_spring_const"] = []
    force_data["keep_pot_v2_distance"] = []
    force_data["keep_pot_v2_fragm1"] = []
    force_data["keep_pot_v2_fragm2"] = []
    
    for i in range(int(len(args.keep_pot_v2)/4)):
        force_data["keep_pot_v2_spring_const"].append(float(args.keep_pot_v2[4*i]))#au
        force_data["keep_pot_v2_distance"].append(float(args.keep_pot_v2[4*i+1]))#ang
        force_data["keep_pot_v2_fragm1"].append(num_parse(args.keep_pot_v2[4*i+2]))
        force_data["keep_pot_v2_fragm2"].append(num_parse(args.keep_pot_v2[4*i+3]))
        
        
    #---------------------
    if len(args.keep_angle) % 3 != 0:
        print("invaild input (-ka)")
        sys.exit(0)
    
    force_data["keep_angle_spring_const"] = []
    force_data["keep_angle_angle"] = []
    force_data["keep_angle_atom_pairs"] = []
    
    for i in range(int(len(args.keep_angle)/3)):
        force_data["keep_angle_spring_const"].append(float(args.keep_angle[3*i]))#au
        force_data["keep_angle_angle"].append(float(args.keep_angle[3*i+1]))#degrees
        force_data["keep_angle_atom_pairs"].append(num_parse(args.keep_angle[3*i+2]))
        if len(force_data["keep_angle_atom_pairs"][i]) != 3:
            print("invaild input (-ka atom_pairs)")
            sys.exit(0)
    
    #---------------------
    if len(args.keep_angle_v2) % 5 != 0:
        print("invaild input (-kav2)")
        sys.exit(0)
    
    force_data["keep_angle_v2_spring_const"] = []
    force_data["keep_angle_v2_angle"] = []
    force_data["keep_angle_v2_fragm1"] = []
    force_data["keep_angle_v2_fragm2"] = []
    force_data["keep_angle_v2_fragm3"] = []
    
    for i in range(int(len(args.keep_angle_v2)/5)):
        force_data["keep_angle_v2_spring_const"].append(float(args.keep_angle_v2[5*i]))#au
        force_data["keep_angle_v2_angle"].append(float(args.keep_angle_v2[5*i+1]))#degrees
        force_data["keep_angle_v2_fragm1"].append(num_parse(args.keep_angle_v2[5*i+2]))
        force_data["keep_angle_v2_fragm2"].append(num_parse(args.keep_angle_v2[5*i+3]))
        force_data["keep_angle_v2_fragm3"].append(num_parse(args.keep_angle_v2[5*i+4]))
       
    #---------------------
    
    if len(args.atom_distance_dependent_keep_angle) % 7 != 0:#[[spring const.(a.u.)] [minimum keep angle (degrees)] [maximum keep angle (degrees)] [base distance (ang.)] [reference atom (1 atom)] [center atom (1 atom)] [atom1,atom2,atom3] ...]
        print("invaild input (-ddka)")
        sys.exit(0)
    
    force_data["aDD_keep_angle_spring_const"] = []
    force_data["aDD_keep_angle_min_angle"] = []
    force_data["aDD_keep_angle_max_angle"] = []
    force_data["aDD_keep_angle_base_dist"] = []
    force_data["aDD_keep_angle_reference_atom"] = []
    force_data["aDD_keep_angle_center_atom"] = []
    force_data["aDD_keep_angle_atoms"] = []
    
    for i in range(int(len(args.atom_distance_dependent_keep_angle)/7)):
        force_data["aDD_keep_angle_spring_const"].append(float(args.atom_distance_dependent_keep_angle[7*i]))#au
        force_data["aDD_keep_angle_min_angle"].append(float(args.atom_distance_dependent_keep_angle[7*i+1]))#degrees
        force_data["aDD_keep_angle_max_angle"].append(float(args.atom_distance_dependent_keep_angle[7*i+2]))#degrees
        if float(args.atom_distance_dependent_keep_angle[7*i+1]) > float(args.atom_distance_dependent_keep_angle[7*i+2]):
            print("invaild input (-ddka min_angle > max_angle)")
            sys.exit(0)
        
        force_data["aDD_keep_angle_base_dist"].append(float(args.atom_distance_dependent_keep_angle[7*i+3]))#ang.
        force_data["aDD_keep_angle_reference_atom"].append(int(args.atom_distance_dependent_keep_angle[7*i+4]))#ang.
        force_data["aDD_keep_angle_center_atom"].append(int(args.atom_distance_dependent_keep_angle[7*i+5]))#ang.
        force_data["aDD_keep_angle_atoms"].append(num_parse(args.atom_distance_dependent_keep_angle[7*i+6]))
        if len(force_data["aDD_keep_angle_atoms"][i]) != 3:
            print("invaild input (-ddka atoms)")
            sys.exit(0)
    #---------------------
    if len(args.keep_dihedral_angle) % 3 != 0:
        print("invaild input (-kda)")
        sys.exit(0)
        
    force_data["keep_dihedral_angle_spring_const"] = []
    force_data["keep_dihedral_angle_angle"] = []
    force_data["keep_dihedral_angle_atom_pairs"] = []
    
    for i in range(int(len(args.keep_dihedral_angle)/3)):
        force_data["keep_dihedral_angle_spring_const"].append(float(args.keep_dihedral_angle[3*i]))#au
        force_data["keep_dihedral_angle_angle"].append(float(args.keep_dihedral_angle[3*i+1]))#degrees
        force_data["keep_dihedral_angle_atom_pairs"].append(num_parse(args.keep_dihedral_angle[3*i+2]))
        if len(force_data["keep_dihedral_angle_atom_pairs"][i]) != 4:
            print("invaild input (-kda atom_pairs)")
            sys.exit(0)
    
    #---------------------
    if len(args.keep_dihedral_angle_v2) % 6 != 0:
        print("invaild input (-kdav2)")
        sys.exit(0)
        
    force_data["keep_dihedral_angle_v2_spring_const"] = []
    force_data["keep_dihedral_angle_v2_angle"] = []
    force_data["keep_dihedral_angle_v2_fragm1"] = []
    force_data["keep_dihedral_angle_v2_fragm2"] = []
    force_data["keep_dihedral_angle_v2_fragm3"] = []
    force_data["keep_dihedral_angle_v2_fragm4"] = []
    
    for i in range(int(len(args.keep_dihedral_angle_v2)/6)):
        force_data["keep_dihedral_angle_v2_spring_const"].append(float(args.keep_dihedral_angle_v2[6*i]))#au
        force_data["keep_dihedral_angle_v2_angle"].append(float(args.keep_dihedral_angle_v2[6*i+1]))#degrees
        force_data["keep_dihedral_angle_v2_fragm1"].append(num_parse(args.keep_dihedral_angle_v2[6*i+2]))
        force_data["keep_dihedral_angle_v2_fragm2"].append(num_parse(args.keep_dihedral_angle_v2[6*i+3]))
        force_data["keep_dihedral_angle_v2_fragm3"].append(num_parse(args.keep_dihedral_angle_v2[6*i+4]))
        force_data["keep_dihedral_angle_v2_fragm4"].append(num_parse(args.keep_dihedral_angle_v2[6*i+5]))
        
    
    #---------------------
    if len(args.well_pot) % 4 != 0:
        print("invaild input (-wp)")
        sys.exit(0)
        
    force_data["well_pot_wall_energy"] = []
    force_data["well_pot_fragm_1"] = []
    force_data["well_pot_fragm_2"] = []
    force_data["well_pot_limit_dist"] = []
    
    for i in range(int(len(args.well_pot)/4)):
        force_data["well_pot_wall_energy"].append(float(args.well_pot[4*i]))#kJ/mol
        force_data["well_pot_fragm_1"].append(num_parse(args.well_pot[4*i+1]))
        force_data["well_pot_fragm_2"].append(num_parse(args.well_pot[4*i+2]))
        force_data["well_pot_limit_dist"].append(args.well_pot[4*i+3].split(","))#ang
        if float(force_data["well_pot_limit_dist"][i][0]) < float(force_data["well_pot_limit_dist"][i][1]) and float(force_data["well_pot_limit_dist"][i][1]) < float(force_data["well_pot_limit_dist"][i][2]) and float(force_data["well_pot_limit_dist"][i][2]) < float(force_data["well_pot_limit_dist"][i][3]):
            pass
        else:
            print("invaild input (-wp a<b<c<d)")
            sys.exit(0)
            
    #---------------------
    if len(args.wall_well_pot) % 4 != 0:
        print("invaild input (-wwp)")
        sys.exit(0)
        
    force_data["wall_well_pot_wall_energy"] = []
    force_data["wall_well_pot_direction"] = []
    force_data["wall_well_pot_limit_dist"] = []
    force_data["wall_well_pot_target"] = []
    
    for i in range(int(len(args.wall_well_pot)/4)):
        force_data["wall_well_pot_wall_energy"].append(float(args.wall_well_pot[4*i]))#kJ/mol
        force_data["wall_well_pot_direction"].append(args.wall_well_pot[4*i+1])
        
        if force_data["wall_well_pot_direction"][i] == "x" or force_data["wall_well_pot_direction"][i] == "y" or force_data["wall_well_pot_direction"][i] == "z":
            pass
        else:
            print("invaild input (-wwp direction)")
            sys.exit(0)
        
        force_data["wall_well_pot_limit_dist"].append(args.wall_well_pot[4*i+2].split(","))#ang
        if float(force_data["wall_well_pot_limit_dist"][i][0]) < float(force_data["wall_well_pot_limit_dist"][i][1]) and float(force_data["wall_well_pot_limit_dist"][i][1]) < float(force_data["wall_well_pot_limit_dist"][i][2]) and float(force_data["wall_well_pot_limit_dist"][i][2]) < float(force_data["wall_well_pot_limit_dist"][i][3]):
            pass
        else:
            print("invaild input (-wwp a<b<c<d)")
            sys.exit(0)
        
        force_data["wall_well_pot_target"].append(num_parse(args.wall_well_pot[4*i+3]))
    #---------------------
    
    if len(args.void_point_well_pot) % 4 != 0:
        print("invaild input (-vpwp)")
        sys.exit(0)
        
    force_data["void_point_well_pot_wall_energy"] = []
    force_data["void_point_well_pot_coordinate"] = []
    force_data["void_point_well_pot_limit_dist"] = []
    force_data["void_point_well_pot_target"] = []
    
    for i in range(int(len(args.void_point_well_pot)/4)):
        force_data["void_point_well_pot_wall_energy"].append(float(args.void_point_well_pot[4*i]))#kJ/mol
        
        
        force_data["void_point_well_pot_coordinate"].append(list(map(float, args.void_point_well_pot[4*i+1].split(","))))
        
        if len(force_data["void_point_well_pot_coordinate"][i]) != 3:
            print("invaild input (-vpwp coordinate)")
            sys.exit(0)
        
        force_data["void_point_well_pot_limit_dist"].append(args.void_point_well_pot[4*i+2].split(","))#ang
        if float(force_data["void_point_well_pot_limit_dist"][i][0]) < float(force_data["void_point_well_pot_limit_dist"][i][1]) and float(force_data["void_point_well_pot_limit_dist"][i][1]) < float(force_data["void_point_well_pot_limit_dist"][i][2]) and float(force_data["void_point_well_pot_limit_dist"][i][2]) < float(force_data["void_point_well_pot_limit_dist"][i][3]):
            pass
        else:
            print("invaild input (-vpwp a<b<c<d)")
            sys.exit(0)
            
        force_data["void_point_well_pot_target"].append(num_parse(args.void_point_well_pot[4*i+3]))
        
    #---------------------
    
    if len(args.around_well_pot) % 4 != 0:
        print("invaild input (-awp)")
        sys.exit(0)
        
    force_data["around_well_pot_wall_energy"] = []
    force_data["around_well_pot_center"] = []
    force_data["around_well_pot_limit_dist"] = []
    force_data["around_well_pot_target"] = []
    
    for i in range(int(len(args.around_well_pot)/4)):
        force_data["around_well_pot_wall_energy"].append(float(args.around_well_pot[4*i]))#kJ/mol
        
        
        force_data["around_well_pot_center"].append(num_parse(args.around_well_pot[4*i+1]))
        
        
        force_data["around_well_pot_limit_dist"].append(args.around_well_pot[4*i+2].split(","))#ang
        if float(force_data["around_well_pot_limit_dist"][i][0]) < float(force_data["around_well_pot_limit_dist"][i][1]) and float(force_data["around_well_pot_limit_dist"][i][1]) < float(force_data["around_well_pot_limit_dist"][i][2]) and float(force_data["around_well_pot_limit_dist"][i][2]) < float(force_data["around_well_pot_limit_dist"][i][3]):
            pass
        else:
            print("invaild input (-vpwp a<b<c<d)")
            sys.exit(0)
            
        force_data["around_well_pot_target"].append(num_parse(args.around_well_pot[4*i+3]))
        
    #---------------------
    
    if len(args.void_point_pot) % 5 != 0:
        print("invaild input (-vpp)")
        sys.exit(0)
    
    force_data["void_point_pot_spring_const"] = []
    force_data["void_point_pot_distance"] = []
    force_data["void_point_pot_coord"] = []
    force_data["void_point_pot_atoms"] = []
    force_data["void_point_pot_order"] = []
    
    for i in range(int(len(args.void_point_pot)/5)):
        force_data["void_point_pot_spring_const"].append(float(args.void_point_pot[5*i]))#au
        force_data["void_point_pot_distance"].append(float(args.void_point_pot[5*i+1]))#ang
        coord = args.void_point_pot[5*i+2].split(",")
        force_data["void_point_pot_coord"].append(list(map(float, coord)))#ang
        force_data["void_point_pot_atoms"].append(num_parse(args.void_point_pot[5*i+3]))
        force_data["void_point_pot_order"].append(float(args.void_point_pot[5*i+4]))
    #---------------------

    
    
    if len(args.fix_atoms) > 0:
        force_data["fix_atoms"] = num_parse(args.fix_atoms[0])
    else:
        force_data["fix_atoms"] = ""
    
    if len(args.gradient_fix_atoms) > 0:
        force_data["gradient_fix_atoms"] = []
        
        for j in range(len(args.gradient_fix_atoms)):
           
            force_data["gradient_fix_atoms"].append(num_parse(args.gradient_fix_atoms[j]))
    else:
        force_data["gradient_fix_atoms"] = ""
    
    force_data["geom_info"] = num_parse(args.geom_info[0])
    
    force_data["opt_method"] = args.opt_method
    
    force_data["xtb"] = args.usextb
    force_data["opt_fragment"] = [num_parse(args.opt_fragment[i]) for i in range(len(args.opt_fragment))]
    return force_data


class BiasPotInterface:
    def __init__(self):
        self.manual_AFIR = ['0.0', '1', '2'] #manual-AFIR (ex.) [[Gamma(kJ/mol)] [Fragm.1(ex. 1,2,3-5)] [Fragm.2] ...]
        self.repulsive_potential = ['0.0','1.0', '1', '2', 'scale'] #Add LJ repulsive_potential based on UFF (ex.) [[well_scale] [dist_scale] [Fragm.1(ex. 1,2,3-5)] [Fragm.2] [scale or value (ang. kJ/mol)] ...]
        self.repulsive_potential_v2 = ['0.0','1.0','0.0','1','2','12','6', '1,2', '1-2', 'scale']#Add LJ repulsive_potential based on UFF (ver.2) (eq. V = ε[A * (σ/r)^(rep) - B * (σ/r)^(attr)]) (ex.) [[well_scale] [dist_scale] [length (ang.)] [const. (rep)] [const. (attr)] [order (rep)] [order (attr)] [LJ center atom (1,2)] [target atoms (3-5,8)] [scale or value (ang. kJ/mol)] ...]

        self.cone_potential = ['0.0','1.0','90','1', '2,3,4', '5-9']#'Add cone type LJ repulsive_potential based on UFF (ex.) [[well_value (epsilon) (kJ/mol)] [dist (sigma) (ang.)] [cone angle (deg.)] [LJ center atom (1)] [three atoms (2,3,4) ] [target atoms (5-9)] ...]')
        
        self.keep_pot = ['0.0', '1.0', '1,2']#keep potential 0.5*k*(r - r0)^2 (ex.) [[spring const.(a.u.)] [keep distance (ang.)] [atom1,atom2] ...] 
        self.keep_pot_v2 = ['0.0', '1.0', '1','2']#keep potential 0.5*k*(r - r0)^2 (ex.) [[spring const.(a.u.)] [keep distance (ang.)] [atom1,atom2] ...] 
        self.anharmonic_keep_pot = ['0.0', '1.0', '1.0', '1,2']#Morse potential  De*[1-exp(-((k/2*De)^0.5)*(r - r0))]^2 (ex.) [[potential well depth (a.u.)] [spring const.(a.u.)] [keep distance (ang.)] [atom1,atom2] ...] 
        self.keep_angle = ['0.0', '90', '1,2,3']#keep angle 0.5*k*(θ - θ0)^2 (0 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep angle (degrees)] [atom1,atom2,atom3] ...] 
        self.keep_angle_v2 = ['0.0', '90', '1','2','3']#keep angle 0.5*k*(θ - θ0)^2 (0 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep angle (degrees)] [atom1,atom2,atom3] ...] 
        self.atom_distance_dependent_keep_angle = ['0.0', '90', "120", "1.4", "5", "1", '2,3,4']#'atom-distance-dependent keep angle (ex.) [[spring const.(a.u.)] [minimum keep angle (degrees)] [maximum keep angle (degrees)] [base distance (ang.)] [reference atom (1 atom)] [center atom (1 atom)] [atom1,atom2,atom3] ...] '
        
        self.keep_dihedral_angle = ['0.0', '90', '1,2,3,4']#keep dihedral angle 0.5*k*(φ - φ0)^2 (-180 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep dihedral angle (degrees)] [atom1,atom2,atom3,atom4] ...] 
        self.keep_dihedral_angle_v2 = ['0.0', '90', '1','2','3','4']#keep dihedral angle 0.5*k*(φ - φ0)^2 (-180 ~ 180 deg.) (ex.) [[spring const.(a.u.)] [keep dihedral angle (degrees)] [atom1,atom2,atom3,atom4] ...] 
        self.void_point_pot = ['0.0', '1.0', '0.0,0.0,0.0', '1',"2.0"]#void point keep potential (ex.) [[spring const.(a.u.)] [keep distance (ang.)] [void_point (x,y,z) (ang.)] [atoms(ex. 1,2,3-5)] [order p "(1/p)*k*(r - r0)^p"] ...] 

        self.well_pot = ['0.0','1','2','0.5,0.6,1.5,1.6']
        self.wall_well_pot = ['0.0','x','0.5,0.6,1.5,1.6', '1']#Add potential to limit atoms movement. (sandwich) (ex.) [[wall energy (kJ/mol)] [direction (x,y,z)] [a,b,c,d (a<b<c<d) (ang.)] [target atoms (1,2,3-5)] ...]")
        self.void_point_well_pot = ['0.0','0.0,0.0,0.0','0.5,0.6,1.5,1.6', '1']#"Add potential to limit atom movement. (sphere) (ex.) [[wall energy (kJ/mol)] [coordinate (x,y,z) (ang.)] [a,b,c,d (a<b<c<d) (ang.)] [target atoms (1,2,3-5)] ...]")
        self.around_well_pot =['0.0','1','0.5,0.6,1.5,1.6',"2"] #Add potential to limit atom movement. (like sphere around 1 atom) (ex.) [[wall energy (kJ/mol)] [1 atom (1)] [a,b,c,d (a<b<c<d) (ang.)]  [target atoms (2,3-5)] ...]")


class iEIPInterface(BiasPotInterface):# inheritance is not good for readable code.
    def __init__(self, folder_name=""):
        super().__init__()
        self.INPUT = folder_name
        self.basisset = '6-31G(d)'#basisset (ex. 6-31G*)
        self.functional = 'b3lyp'#functional(ex. b3lyp)
        self.sub_basisset = '' #sub_basisset (ex. I LanL2DZ)

        self.N_THREAD = 8 #threads
        self.SET_MEMORY = '1GB' #use mem(ex. 1GB)
 
        self.usextb = "None"#use extended tight bonding method to calculate. default is not using extended tight binding method (ex.) GFN1-xTB, GFN2-xTB 
        
        self.pyscf = False
        self.electronic_charge = 0
        self.spin_multiplicity = 1#'spin multiplcity (if you use pyscf, please input S value (mol.spin = 2S = Nalpha - Nbeta)) (ex.) [multiplcity (0)]'
        
        self.fix_atoms = []  
        self.geom_info = []
        self.opt_method = ""
        self.opt_fragment = []
        return

class NEBInterface(BiasPotInterface):# inheritance is not good for readable code.
    def __init__(self, folder_name=""):
        super().__init__()
        self.INPUT = folder_name
        self.basisset = '6-31G(d)'
        self.functional = 'b3lyp'
        self.NSTEP = "10"
        self.OM = False
        self.partition = "0"
        self.N_THREAD = "8"
        self.SET_MEMORY = "1GB"
        self.apply_CI_NEB = '99999'
        self.steepest_descent = '99999'
        self.usextb = "None"
        self.fix_atoms = []  
        self.geom_info = []
        self.opt_method = ""
        self.opt_fragment = []
        return
    


class OptimizeInterface(BiasPotInterface):# inheritance is not good for readable code.
    def __init__(self, input_file=""):
        super().__init__()
        self.INPUT = input_file
        self.basisset = '6-31G(d)'#basisset (ex. 6-31G*)
        self.functional = 'b3lyp'#functional(ex. b3lyp)
        self.sub_basisset = '' #sub_basisset (ex. I LanL2DZ)

        self.NSTEP = 300 #iter. number
        self.N_THREAD = 8 #threads
        self.SET_MEMORY = '1GB' #use mem(ex. 1GB)
        self.DELTA = 'x'

        self.fix_atoms = ""#fix atoms (ex.) [atoms (ex.) 1,2,3-6]
        self.md_like_perturbation = "0.0"
        self.geom_info = "1"#calculate atom distances, angles, and dihedral angles in every iteration (energy_profile is also saved.) (ex.) [atoms (ex.) 1,2,3-6]
        self.opt_method = ["AdaBelief"]#optimization method for QM calclation (default: AdaBelief) (mehod_list:(steepest descent method) RADAM, AdaBelief, AdaDiff, EVE, AdamW, Adam, Adadelta, Adafactor, Prodigy, NAdam, AdaMax, FIRE third_order_momentum_Adam (quasi-Newton method) mBFGS, mFSB, RFO_mBFGS, RFO_mFSB, FSB, RFO_FSB, BFGS, RFO_BFGS, TRM_FSB, TRM_BFGS) (notice you can combine two methods, steepest descent family and quasi-Newton method family. The later method is used if gradient is small enough. [[steepest descent] [quasi-Newton method]]) (ex.) [opt_method]
        self.calc_exact_hess = -1#calculate exact hessian per steps (ex.) [steps per one hess calculation]
        self.usextb = "None"#use extended tight bonding method to calculate. default is not using extended tight binding method (ex.) GFN1-xTB, GFN2-xTB 
        self.DS_AFIR = False
        self.pyscf = False
        self.electronic_charge = 0
        self.spin_multiplicity = 1#'spin multiplcity (if you use pyscf, please input S value (mol.spin = 2S = Nalpha - Nbeta)) (ex.) [multiplcity (0)]'
        self.saddle_order = 0
        self.opt_fragment = []
        return
 
