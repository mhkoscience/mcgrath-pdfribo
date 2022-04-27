#!/usr/bin/python3

# https://github.com/mhkoscience/mcgrath-pdfribo

##############################################################################
# some setup
##############################################################################

b = "../input"
bd_list = [f"{b}/PDF_plus",
           f"{b}/PDF_minus"]

selection_name = "uL22_and_below_larger"
train_inds = "01"
percentage_variance_covered = 0.95
n_pcs_for_ani = 5


##############################################################################
# program body
##############################################################################

import sys
import numpy as np
import MDAnalysis as mda
import MDAnalysis.analysis.pca as mda_pca
from pathlib import Path

# append directory which contains analysis_modules to path
sys.path.append("../modules")

import FMA_functions as fma
import PCA_functions as pca


# test indices
test_inds = [i for i in "0123" if i not in train_inds ]


# input file names
aligned_pdb = f"{bd_list[0]}/T0/{selection_name}.pdb"
aligned_xtc = f"{bd_list[0]}/T0/{selection_name}.xtc"

uni = mda.Universe(aligned_pdb, aligned_xtc)
traj_len = len(uni.trajectory)
print(f"traj_len = {traj_len:.0f}")

# make binary function arrays
f_train = np.append(np.ones(traj_len * len(train_inds)),
                    np.zeros(traj_len * len(train_inds)))
f_test = np.append(np.ones(traj_len * len(test_inds)),
                   np.zeros(traj_len * len(test_inds)))

print(f"f_train.shape = {np.shape(f_train)}")
print(f"f_test.shape = {np.shape(f_test)}")

# create lists for trajectory joining
train_trj_list = []
test_trj_list = []

for i in range(len(bd_list)):

    for ind in train_inds:
        train_trj_list.append(f"{bd_list[i]}/T{ind}/{selection_name}.xtc")

    for ind in test_inds:
        test_trj_list.append(f"{bd_list[i]}/T{ind}/{selection_name}.xtc")

print("train_trj_list")
print(train_trj_list)
print("test_trj_list")
print(test_trj_list)

# create aligned training and test universes
aligned_u_train = mda.Universe(f"{bd_list[0]}/T0/{selection_name}.pdb",
                               train_trj_list)
aligned_u_test = mda.Universe(f"{bd_list[0]}/T0/{selection_name}.pdb",
                              test_trj_list)

# perform PCA and generate pc_object
pc_object = mda_pca.PCA(aligned_u_train).run()

# reduce dimensionality
n_pcs = np.where(pc_object.cumulated_variance > percentage_variance_covered)[0][0]
print(f"n_pcs = {n_pcs}")
pcs_subset = pc_object.p_components[:, :n_pcs]

# calculate vector a which points in the direction of MCM
# also return alpha, the vector of weights of individual PCs
alpha, beta, a = fma.get_alpha_beta_and_a(pc_object, aligned_u_train, f_train,
                                          n_pcs)


# handle directories
tr_dir = "".join([str(x) for x in train_inds])
te_dir = "".join([str(x) for x in test_inds])
ani_dir = f"{b}/../output_pdb/train{tr_dir}_test{te_dir}/{selection_name}"
csv_dir = f"{b}/../output_csv/train{tr_dir}_test{te_dir}/{selection_name}"
Path(ani_dir).mkdir(exist_ok=True, parents=True)
Path(csv_dir).mkdir(exist_ok=True, parents=True)

# ew coefficients to be compared with alpha
ew_coefficients = fma.get_ew_coefficients(pc_object, alpha, n_pcs)

pci_ani_path = f"{ani_dir}/{selection_name}"
mcm_ani_path = f"{ani_dir}/{selection_name}_mcm"
ew_mcm_ani_path = f"{ani_dir}/{selection_name}_ew_mcm"


# generate animation .pdb and .xtc files
pca.make_pci_ani(pc_object, aligned_u_train, n_pcs_for_ani, pci_ani_path)
fma.make_mcm_ani(aligned_u_train, a, mcm_ani_path)
fma.make_ew_mcm_ani(pc_object, aligned_u_train, alpha, a, n_pcs,
                    ew_mcm_ani_path)


# generate data for model validation
f_model_train = fma.get_f_model(pc_object, aligned_u_train, beta, f_train,
                                n_pcs)
f_model_test = fma.get_f_model(pc_object, aligned_u_test, beta, f_train, n_pcs)
rm, rc = fma.get_rm_and_rc(aligned_u_train, a, f_model_test, f_train, f_test)

mse_train = np.mean((f_model_train - f_train) ** 2)
mse_test = np.mean((f_model_test - f_test) ** 2)


# save data needed for graphs to .csv files
np.savetxt(f"{csv_dir}/var_cvar.csv.gz",
           np.array([pc_object.variance, pc_object.cumulated_variance]).T,
           header="variance,cvariance",
           fmt="%.4e",
           comments="",
           delimiter=",")
np.savetxt(f"{csv_dir}/alpha_beta_ew_coefficients.csv.gz",
           np.array([alpha, beta, ew_coefficients]).T,
           header="alpha,beta,ew_coeff",
           fmt="%.4e",
           comments="",
           delimiter=",")
np.savetxt(f"{csv_dir}/a.csv.gz",
           np.array([a]).T,
           header="ai",
           fmt="%.4e",
           comments="",
           delimiter=",")
np.savetxt(f"{csv_dir}/f_test_model.csv.gz",
           np.array([f_test, f_model_test]).T,
           header="f, fm",
           fmt="%.4e",
           comments="",
           delimiter=",")
np.savetxt(f"{csv_dir}/f_train_model.csv.gz",
           np.array([f_train, f_model_train]).T,
           header="f, fm",
           fmt="%.4e",
           comments="",
           delimiter=",")
np.savetxt(f"{csv_dir}/rm_rc.csv", np.array([rm, rc]))
np.savetxt(f"{csv_dir}/mse_train_test.csv", np.array([mse_train, mse_test]))
np.savetxt(f"{csv_dir}/pcs_subset.csv.gz", pcs_subset.T,
           fmt="%.4e",
           comments="",
           delimiter=",")
