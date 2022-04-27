"""FMA helps us find the collective motion of biomolecules correlated to
a scalar functional quantity of interest. Detailed information about the
method can be found in the following article:

Detection of Functional Modes in Protein Dynamics
J. S. Hub, B. L. de Groot
doi:10.1371/journal.pcbi.1000480

Side note: Notation regarding PCA in the following code is different
from the notation used in the article. The article denotes the vectors
pointing in the direction of maximal variance e_i and the projections
onto these vectors as PC_i. In the following code and comments I denote
the vector pointing in the direction of maximal variance as
PC - principal component and I do not use a special term for the
projections onto these vectors. This notation should be consistent with
that used in the MDAnalysis documentation. """

import MDAnalysis as mda
import numpy as np


def _extract_trj(aligned_u):
    """
    Extract the trajectory from the universe object into a numpy array
    and calculate the mean structure of this trajectory.

    Parameters:

        aligned_u (object): Universe object created from the
        aligned trajectories.

    Returns:

        xyz_t (array (n_frames, 3 * n_atoms)): Trajectory extracted into
        an array.
    """

    trj = aligned_u.trajectory
    atomgroup = aligned_u.select_atoms("all")

    n_frames = len(trj)
    n_atoms = len(aligned_u.atoms)

    # initialize empty array
    xyz_t = np.empty((n_frames, 3 * n_atoms))

    # fill array, for every frame insert structure coordinates
    # ts is just to loop through the trajectory, frame is the actual
    # frame number
    for frame, ts in enumerate(trj):
        xyz_t[frame, :] = atomgroup.positions.ravel()

    return xyz_t


def _get_projs(xyz_t, a):
    """
    Calculate projections of trajectory onto MCM vector a.

    Parameters:

        xyz_t (array (n_frames, 3 * n_atoms)): Trajectory extracted into
        an array.

        a (array (3 * n_atoms, )): The vector pointing in the direction
        of MCM.

    Returns:

        pa_t (array (n_frames, )): Projections onto the vector a.

    """

    n_frames = xyz_t.shape[0]

    # average over time
    xyz_mean = np.mean(xyz_t, axis=0)
    xyz_t_centered = np.subtract(xyz_t, xyz_mean)

    # initialize empty array
    pa_t = np.empty((n_frames,))

    for frame in range(n_frames):
        # fill projs array
        # project the trajectory onto the vector a
        pa_t[frame] = np.dot(a, xyz_t_centered[frame, :])

    return pa_t


def _get_param_values(pa_t, n_ani_frames):
    """
    Calculate parameter values for creating animations.

        Parameters:

            pa_t (array (n_frames, )): Projections onto the vector a.

            n_ani_frames (int): Number of frames for animation.

        Returns:

            param_values (array (len(param_values, )): Parameter
            values used to create animations of MCM and ewMCM.

        """

    # calculate parameter values by creating an equally spaced array
    # from min to max of the projections onto a
    param_values = np.linspace(min(pa_t), max(pa_t), n_ani_frames)

    return param_values


def get_alpha_beta_and_a(pc_object, aligned_u_train, f_train, n_pcs,
                         mutual_information=False):
    """
    Calculates the vector a which point in the direction of MCM.

    Parameters:

        pc_object (object): MDAnalysis object which contains all
        relevant outputs from PCA.

        aligned_u_train (object): Universe object created from
        the aligned trajectories. Training subset.

        f_train (array (n_frames_train, )):   The functional
        quantity related to the motion of the biomolecule. Training
        subset.

        n_pcs (int): Number of PCs describing a large enough percentage
        of total variance.

        mutual_information (bool, optional, default=False): If True, use mutual
        information as a measure of correlation of f and pa.

    Returns:

        alpha (array (n_pcs, )): Coefficients representing the weights
        of PCs in the vector a (normalized).

        beta (array (n_pcs, )): Coefficients representing the weights
        of PCs in the vector a (not normalized).

        a (array (3 * n_atoms, )): The vector pointing in the direction
        of MCM.
    """

    # project the selection onto a subset of PCs
    projs_subset = pc_object.transform(
        aligned_u_train.select_atoms("all"), n_components=n_pcs)

    # extract principal component array from pc_object
    pcs_subset = pc_object.p_components[:, :n_pcs]

    # calculate the covariance matrix of the projections:
    # C_ij = cov(p_i, p_j)
    C = np.cov(projs_subset.T)

    # covariance vector of projections onto PCs and the functional
    # quantity f
    cov_f = np.empty((n_pcs,))
    for pc in range(n_pcs):
        cov_f[pc] = np.cov(projs_subset.T[pc], f_train)[0, 1]

    # equation 5 in the article
    beta = np.linalg.solve(C, cov_f)
    alpha = beta / np.linalg.norm(beta)
    a = np.matmul(pcs_subset, alpha)

    if mutual_information:

        a0 = a
        alpha0 = alpha

    return alpha, beta, a


def make_mcm_ani(aligned_u_train, a, ani_path, n_ani_frames=100):
    """
    Creates an animation of MCM. Sometimes this motion is very
    indistinct when vector a points in the direction of a barrier in the
    energy landscape (see figure 1 in the article).
    
    Parameters:
        
        aligned_u_train (object): Universe object created from
        the aligned trajectories. Training subset.
        
        a (array (3 * n_atoms, )): The vector pointing in the direction
        of MCM.
        
        ani_path (string): Path to and name of the animation
        without file extension.

        n_ani_frames (int): Number of frames for animation.
    """

    # extract trajectory into array
    xyz_t = _extract_trj(aligned_u_train)
    # average over time
    xyz_mean = np.mean(xyz_t, axis=0)
    pa_t = _get_projs(xyz_t, a)
    # calculate parameter values
    param_values = _get_param_values(pa_t, n_ani_frames)
    n_atoms = len(aligned_u_train.atoms)

    # initialize empty array
    mcm = np.empty((n_ani_frames, n_atoms, 3))

    for frame in range(n_ani_frames):
        # fill animation trajectory array
        # add vector a multiplied by a changing parameter to an average
        # structure
        # also reshape to visualise in 3D space
        mcm[frame, :, :] = np.reshape(xyz_mean + a * param_values[frame],
                                      (n_atoms, 3))

    ref_selection = aligned_u_train.select_atoms("all")
    # create universe which has all the information about the topology
    # of the system
    ani_u = mda.Merge(ref_selection)
    # load in the animation trajectory from an array
    # fac means frames, atoms, coordinates (order of the array
    # dimensions)
    ani_u = ani_u.load_new(mcm, order="fac")
    ani_selection = ani_u.select_atoms("all")
    # write it out into .pdb and .xtc files so it can be visualized in
    # VMD
    ani_selection.write(f"{ani_path}.pdb")
    ani_selection.write(f"{ani_path}.xtc", frames="all")


def get_ew_coefficients(pc_object, alpha, n_pcs):
    """
    A function to help simplify calculating the ewMCM but also needed to
    return the coefficients which can later be plotted to compare with
    alpha.

    Parameters:

        pc_object (object): MDAnalysis object which contains all
        relevant outputs from PCA.

        alpha (array (n_pcs, )): Coefficients representing the weights
        of PCs in the vector a (normalized).

        n_pcs (int): Number of PCs describing a large enough percentage
        of total variance.

    Returns:

        ew_coefficients (array (n_pcs, )): A set of coefficients to
        rescale MCM to ewMCM.
    """

    # extract variance from pc_object
    variance_subset = pc_object.variance[:n_pcs]

    # preparing variables for equation 12 to make it simpler
    alpha_var_elementwise = np.multiply(alpha, variance_subset)
    alpha_sd_elementwise = np.multiply(alpha, np.sqrt(variance_subset))

    # initialize empty array
    ew_coefficients = np.empty((n_pcs,))

    for pc in range(n_pcs):
        # part of equation 12 in the article
        ew_coefficients[pc] = (alpha_var_elementwise[pc]) / (
                np.sum(alpha_sd_elementwise ** 2))

    return ew_coefficients


def make_ew_mcm_ani(pc_object, aligned_u_train, alpha, a, n_pcs, ani_path,
                    n_ani_frames=100):
    """
    Creates an animation of ewMCM, it is a rescaled version of MCM which
    takes into account the probability of the motion.

    Parameters:

        pc_object (object): MDAnalysis object which contains all
        relevant outputs from PCA.

        aligned_u_train (object): Universe object created from
        the aligned trajectories. Training subset.

        alpha (array (n_pcs, )): Coefficients representing the weights
        of PCs in the vector a (normalized).

        a (array (3 * n_atoms, )): The vector pointing in the direction
        of MCM.

        n_pcs (int): Number of PCs describing a large enough percentage
        of total variance.

        ani_path (string): Path to and name of the animation
        without file extension.

        n_ani_frames (int): Number of frames for animation.
    """

    # extract trajectory into array
    xyz_t = _extract_trj(aligned_u_train)
    # average over time
    xyz_mean = np.mean(xyz_t, axis=0)

    n_atoms = int(np.shape(xyz_mean)[0] / 3)

    pa_t = _get_projs(xyz_t, a)
    # calculate parameter values
    param_values = _get_param_values(pa_t, n_ani_frames)

    # extract principal component array from pc_object
    pcs_subset = pc_object.p_components[:, :n_pcs]

    ew_coefficients = get_ew_coefficients(pc_object, alpha, n_pcs)

    # initialize empty array
    ew_params = np.empty((n_ani_frames, n_pcs))

    for frame in range(n_ani_frames):

        for pc in range(n_pcs):
            # rescale projections onto the vector a (ensemble weighting)
            # equation 12 in the article
            ew_params[frame, pc] = param_values[frame] * ew_coefficients[pc]

    # reshape mean coordinates back into (n_atoms, 3)
    xyz_mean = np.reshape(xyz_mean, (n_atoms, 3))
    # initialize empty array
    ew_mcm = np.empty((n_ani_frames, n_atoms, 3))

    for frame in range(n_ani_frames):
        # create ewMCM animation trajectory array, necessary to reshape
        # and add to average structure
        # described in the paragraph bellow equation 12
        flat_ew_mcm = np.matmul(pcs_subset, ew_params[frame, :])
        ew_mcm[frame, :, :] = xyz_mean + np.reshape(flat_ew_mcm, (n_atoms, 3))

    ref_selection = aligned_u_train.select_atoms("all")
    # create universe which has all the information about the topology
    # of the system
    ani_u = mda.Merge(ref_selection)
    # load in the animation trajectory from an array
    # fac means frames, atoms, coordinates
    ani_u = ani_u.load_new(ew_mcm, order="fac")
    ani_selection = ani_u.select_atoms("all")
    # write it out into .pdb and .xtc files so it can be visualized in
    # VMD
    ani_selection.write(f"{ani_path}.pdb")
    ani_selection.write(f"{ani_path}.xtc", frames="all")


def get_f_model(pc_object, aligned_u_test, beta, f_train, n_pcs):
    """
    Predicts values of f from test data based on the model created from
    training data.

    Parameters:
        
        pc_object (object): MDAnalysis object which contains all
        relevant outputs from PCA.

        aligned_u_test (object): Universe object created from the
        aligned trajectories. Test subset.
        
        beta (array (n_pcs, )): Coefficients representing the weights
        of PCs in the vector a (not normalized).

        f_train (array (n_frames_train, )):   The functional
        quantity related to the motion of the biomolecule. Training
        subset.
                                                    
        n_pcs (int): Number of PCs describing a large enough percentage
        of total variance.

    Returns:

        f_model (array (n_frames_test, )): Predicted values of f.

    """

    # calculate projections of test universe
    projs_subset = pc_object.transform(aligned_u_test.select_atoms("all"),
                                       n_components=n_pcs)
    projs_centered = projs_subset - np.mean(projs_subset, axis=0)

    f_train_mean = np.mean(f_train)

    # prepare sum of products of beta_i and test projections onto PC_i
    beta_projs_t = np.matmul(projs_centered, beta)

    # equation 6 in the article
    f_model = f_train_mean + beta_projs_t

    return f_model


def get_rm_and_rc(aligned_u_train, a, f_model, f_train, f_test):
    """
    Calculates correlation coefficients R_m and R_c which are used to evaluate
    Parameters:

        aligned_u_train (object): Universe object created from the
        aligned trajectories. Training subset.

        a (array (3 * n_atoms, )): The vector pointing in the direction
        of MCM.

        f_model (array (n_frames_test, )): Predicted values of f.

        f_train (array (n_frames_train, )):   The functional
        quantity related to the motion of the biomolecule. Training
        subset.

        f_test (array (n_frames_test, )):   The functional quantity
        related to the motion of the biomolecule. Test subset.

    Returns:

        rm (float): Correlation coefficient which was maximized in the
        creation of the model (calculation of vector a). Equation 2 in
        the article.

        rc (float): Cross-validation correlation coefficient, quantifies
        correlation of predicted values of f with the real known values
        of f.

    """

    xyz_t = _extract_trj(aligned_u_train)
    pa_t = _get_projs(xyz_t, a)
    # Pearson correlation coefficients R_m and R_c
    rm = np.corrcoef(pa_t, f_train)[1, 0]
    rc = np.corrcoef(f_model, f_test)[1, 0]

    return rm, rc
