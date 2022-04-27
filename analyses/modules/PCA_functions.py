import MDAnalysis as mda
import numpy as np


def make_pci_ani(pc_object, aligned_u, n_pcs_for_ani, ani_path,
                 n_ani_frames=100):
    """
    Creates animations of the motions described by the first
    n_pcs_for_ani principal components.

    Parameters:

        pc_object (object): MDAnalysis object which contains all
        relevant outputs from PCA.

        aligned_u (object): Universe object created from the
        aligned trajectories.

        n_pcs_for_ani (int): Number of PCs for which you want to
        create an animation.

        ani_path (string): Path to and name of the animation
        without file extension. In the name only include the name of the
        selection for which the animation is being created,
        "PC{i}_animation" is added to the name automatically.

        n_ani_frames (int): Number of frames for animation.
    """

    trj = aligned_u.trajectory
    ref_selection = aligned_u.select_atoms("all")
    n_atoms = ref_selection.n_atoms
    n_frames = len(trj)

    projs_subset = pc_object.transform(ref_selection,
                                       n_components=n_pcs_for_ani)

    # initialize empty array
    xyz_t = np.empty((n_frames, 3 * n_atoms))

    # fill array, for every frame insert structure coordinates
    # ts is just to loop through the trajectory, i is the actual frame
    # number
    for frame, ts in enumerate(trj):

        # extract trajectory into array
        xyz_t[frame, :] = ref_selection.positions.ravel()

    # average over time
    xyz_mean = np.mean(xyz_t, axis=0)

    for pc in range(n_pcs_for_ani):

        # calculate parameter values (different for each PC)
        p_i = projs_subset[:, pc]
        param_values = np.linspace(min(p_i), max(p_i), n_ani_frames)

        # extract principal components from pc_object one by one
        pci = pc_object.p_components[:, pc]

        # initialize empty array
        pci_ani_coords = np.empty((n_ani_frames, n_atoms, 3))

        for frame in range(n_ani_frames):

            # add the principal component vector multiplied by a
            # changing scalar parameter to the average structure
            pci_and_coords_flat = xyz_mean + (param_values[frame] * pci)
            pci_ani_coords[frame, :, :] = np.reshape(pci_and_coords_flat,
                                                     (n_atoms, 3))

        # add PC number to the name
        ani_name = f"{ani_path}_PC{pc+1}"

        # create universe which has all the information about the
        # topology of the system
        ani_u = mda.Merge(ref_selection)
        # load in the animation trajectory from an array
        # fac means frames, atoms, coordinates
        ani_u = ani_u.load_new(pci_ani_coords, order="fac")
        ani_selection = ani_u.select_atoms("all")
        # write it out into .pdb and .xtc files so it can be visualized
        # in VMD
        ani_selection.write(f"{ani_name}.pdb")
        ani_selection.write(f"{ani_name}.xtc", frames="all")
