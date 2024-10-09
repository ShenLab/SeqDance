from prody import *
from pylab import *
import numpy as np
import sys, getopt

# Return a force constant based on the squared distance between atoms
def gammaDistanceDependent(dist2, *args):
    if dist2 <= 16:
        return 10
    elif dist2 <= 100:
        return 2
    elif dist2 <= 225:
        return 1
    else:
        return 0

# Classify normal modes into three distinct ranges, each contributing to different dynamics
# The function splits the eigenvalues into ranges that account for 33%, 33-66%, and 66-100% 
# of the total dynamic contribution
def get_three_range_from_eigenvalues(eval):
    # Calculate the fractional contribution of each mode to the overall dynamics 
    # (the contribution is inversely proportional to the eigenvalues)
    contri = (1 / eval) / (1 / eval).sum()
    contri_sum = np.cumsum(contri)

    # Identify mode ranges: 0-33%, 33-66%, and 66-100% of the dynamic contribution
    index_1 = np.argmin(np.abs(contri_sum - 1/3))  # Mode contributing up to 33%
    index_2 = np.argmin(np.abs(contri_sum - 0.5 * (1 + contri_sum[index_1])))  # Adjust for large initial contributions

    mode_ranges = [np.arange(index_1 + 1), np.arange(index_1 + 1, index_2 + 1), np.arange(index_2 + 1, len(eval))]
    return mode_ranges

# Calculate mean square fluctuations (MSFs) and pairwise correlations from eigenvectors and eigenvalues
# MSFs capture residue-level movements, and pairwise correlations reveal how residues move together
def get_msf_and_cor_from_3N_eigen(mode_ranges, evec, eval, N):
    msfs = []  # Store mean square fluctuations for each mode range
    cors = []  # Store correlation matrices for each mode range
    for mode_range in mode_ranges:
        # Normalize eigenvectors to account for the contribution of eigenvalues
        evec_normalize = evec[:, mode_range] / np.sqrt(eval[mode_range])

        # Covariance matrix for all x, y, z coordinates of the Cα atoms
        cov_3n = np.dot(evec_normalize, evec_normalize.T)

        # Reduce the 3Nx3N covariance matrix to NxN by taking the trace (summed over x, y, z axes)
        cov_n = np.trace(cov_3n.reshape((N, 3, N, 3)), axis1=1, axis2=3)

        # Diagonal elements of cov_n give the mean square fluctuations for each residue
        msf = np.diag(cov_n)

        # Normalize the covariance matrix by MSFs to obtain a residue correlation map
        cor_n = cov_n / np.sqrt(np.outer(msf, msf))

        msfs.append(msf)
        cors.append(cor_n)
    return msfs, cors

# Calculate MSFs using the Gaussian Network Model (GNM), which provides isotropic dynamics
def calc_gnm_msf(calphas, N):
    # Perform GNM analysis by building the Kirchhoff matrix
    gnm = GNM('GNM analysis')
    gnm.buildKirchhoff(calphas, cutoff=15, gamma=gammaDistanceDependent)
    gnm.calcModes(N)
    eval = gnm.getEigvals()
    evec = gnm.getEigvecs()

    # Get MSFs for each of the three dynamic ranges
    mode_ranges = get_three_range_from_eigenvalues(eval)
    
    # Calculate MSFs for each normal mode range
    gnm_msf = []
    for mode_range in mode_ranges:
        gnm_msf.append(((evec[:, mode_range] * evec[:, mode_range]) / eval[mode_range]).sum(axis=1))
    return np.array(gnm_msf)

# Calculate pairwise correlations using the Anisotropic Network Model (ANM), which accounts for anisotropic dynamics
def calc_anm_cor(calphas, N):
    # Perform ANM analysis by building the Hessian matrix
    anm = ANM('ANM analysis')
    anm.buildHessian(calphas, cutoff=15, gamma=gammaDistanceDependent)
    anm.calcModes(N * 3)
    eval = anm.getEigvals()
    evec = anm.getEigvecs()

    # Get correlation maps for each of the three dynamic ranges
    mode_ranges = get_three_range_from_eigenvalues(eval)
    
    # Calculate MSFs and correlation maps for each normal mode range
    anm_msf, anm_cor = get_msf_and_cor_from_3N_eigen(mode_ranges, evec, eval, N)
    return np.array(anm_cor)

# Main function to calculate and save NMA-based features from a PDB structure
def main(argv):
    try:
        # Parse command-line arguments for input/output file paths
        opts, args = getopt.getopt(argv, "hi:o:")
    except getopt.GetoptError:
        print('Usage: python NMA_features.py -i <input PDB file> -o <output feature file>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: python NMA_features.py -i <input PDB file> -o <output feature file>')
            sys.exit()
        elif opt in ("-i"):
            pdb_file = arg
        elif opt in ("-o"):
            output_path = f"{arg}.npz"

    # Load the PDB file and select Cα atoms
    ubi = parsePDB(pdb_file)
    calphas = ubi.select('calpha')
    N = len(calphas)

    # Use GNM to calculate residue fluctuations (isotropic movements)
    gnm_msf = calc_gnm_msf(calphas, N)

    # Use ANM to calculate pairwise correlations (anisotropic movements)
    anm_cor = calc_anm_cor(calphas, N)

    # Save the dynamic features to a compressed file
    np.savez_compressed(output_path, gnm_msf=gnm_msf, anm_cor=anm_cor)

    print(f"Finished processing: {pdb_file}")

# Run the script with command-line arguments
if __name__ == "__main__":
    main(sys.argv[1:])