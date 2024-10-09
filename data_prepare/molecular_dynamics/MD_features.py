# Import necessary modules
import math
import pandas as pd
import numpy as np
from sys import stdout
import sys, getopt
import mdtraj  # MDTraj is used to analyze molecular dynamics trajectories

# Function to load the trajectory and PDB, select C-alpha atoms, and align the trajectory to the reference frame
def get_traj_info(traj_file, pdb_file):
    traj = mdtraj.load_dcd(traj_file, pdb_file)  # Load trajectory and PDB
    ca_atoms = traj.topology.select('name CA')  # Select C-alpha atoms
    traj.superpose(traj, 0, atom_indices=ca_atoms)  # Align all frames to the first one using C-alphas
    N = len(ca_atoms)  # Number of C-alpha atoms (residues)
    n_frames = traj.n_frames  # Number of frames in the trajectory

    return traj, ca_atoms, N, n_frames  # Return trajectory and basic information

# Function to compute residue-level features and pairwise correlation matrix
def get_dssp_sasa_angle_rmsf_cor(traj, ca_atoms, N, n_frames):
    # Calculate secondary structures per frame, assign 8 different classes
    dssp = mdtraj.compute_dssp(traj, simplified=False)  
    dssp = pd.DataFrame(dssp)
    # Count occurrences of each secondary structure type across all frames
    dssp_count = pd.DataFrame([dssp[i].value_counts() for i in dssp.columns], columns=['B','E','H','I','G','S','T',' '], index=dssp.columns)
    dssp_count.columns = ['dssp_' + i for i in dssp_count.columns]  # Rename columns for clarity
    
    # Initialize the residue-level feature matrix and normalize DSSP counts by the number of frames
    res_feat = dssp_count.fillna(0) / n_frames
    res_feat.insert(0, 'ca_atom', ca_atoms)  # Add C-alpha atom index
    
    # Calculate solvent accessible surface area (SASA) for each residue, both mean and standard deviation
    sasa = mdtraj.shrake_rupley(traj, mode="residue")
    res_feat['sasa_mean'] = sasa.mean(axis=0)
    res_feat['sasa_std'] = sasa.std(axis=0)
    
    # Calculate angles (phi, psi, chi1) and store angle percentages in 12 bins
    angle_bins = [math.pi * (-1 + i/6) for i in range(0,13)]  # Define bins for angles
    chi1 = mdtraj.compute_chi1(traj)
    phi = mdtraj.compute_phi(traj)
    psi = mdtraj.compute_psi(traj)
    angles = {'chi': chi1, 'phi': phi, 'psi': psi}  # Dictionary for angles
    
    # Loop through each angle type and calculate histogram counts for each residue
    for angle in angles:
        angle_count = []
        for i in range(angles[angle][1].shape[1]):
            counts, bins = np.histogram(angles[angle][1][:,i], bins=angle_bins)  # Calculate histogram
            angle_count.append(counts)
            
        # Convert angle counts into a dataframe and merge with residue feature matrix
        angle_tab = pd.DataFrame(np.array(angle_count), columns=[f'{angle}_{k}' for k in range(12)])
        if angle == 'phi':
            angle_tab.index = angles[angle][0][:,2]  # Use residue index for phi
        else:
            angle_tab.index = angles[angle][0][:,1]  # Use residue index for chi1 and psi
        angle_tab = angle_tab / n_frames  # Normalize by number of frames

        res_feat = pd.merge(res_feat, angle_tab, left_on='ca_atom', right_index=True, how='left')  # Merge with main feature matrix

    # Compute RMSF (Root Mean Square Fluctuation) for each residue
    res_feat['rmsf'] = mdtraj.rmsf(traj, traj, atom_indices=ca_atoms)

    # Calculate pairwise covariance and correlation between C-alpha atoms
    ca_xyz = traj.xyz[:, ca_atoms, :]  # Extract XYZ coordinates for C-alphas
    cov_3n = np.cov(ca_xyz.reshape((-1, len(ca_atoms) * 3)), rowvar=False)  # Calculate covariance matrix
    cov_n = np.trace(cov_3n.reshape((N, 3, N, 3)), axis1=1, axis2=3)  # Trace to get covariance of C-alpha atoms

    # Compute MSF (Mean Square Fluctuation) and normalize to create correlation matrix
    msf = np.diag(cov_n)
    cor_n = cov_n / np.sqrt(np.outer(msf, msf))  # Normalize covariance to get correlation

    return res_feat, cor_n  # Return residue-level features and correlation matrix

# Function to compute pairwise contact maps based on interaction file
def get_contact_maps(inter_file, N, n_frames):
    inter = pd.read_table(inter_file, skiprows=2, header=None)  # Read interaction data from file
    inter.columns = ['frame','interaction','atom1','atom2']
    # Extract residue indices for both interacting residues
    inter['res1'] = inter.atom1.str.split(':', expand=True)[2].astype(int) - 1
    inter['res2'] = inter.atom2.str.split(':', expand=True)[2].astype(int) - 1
    
    # Map interaction types to numeric IDs (9 types of interactions)
    category_mapping = {'vdw':0, 'hbbb':1, 'hbsb':2, 'hbss':3, 'hp':4, 'sb':5, 'pc':6, 'ps':7, 'ts':8}
    inter['interaction_id'] = inter['interaction'].map(category_mapping)  # Assign IDs for interaction types

    # Count the number of frames for each interaction and normalize by the number of frames
    cont_count = inter.groupby(['res1', 'res2', 'interaction_id']).count()[['frame']] / n_frames
    cont_map = np.zeros([N, N, 9])  # Initialize contact map array

    # Fill contact map with interaction counts
    for i in cont_count.index:
        cont_map[i] = cont_count.loc[i]

    return cont_map  # Return contact map

# Main function to handle file I/O and process trajectory data
def main(argv):
    try:
        # Parse command-line arguments for input/output file paths
        opts, args = getopt.getopt(argv, "hp:t:i:o:")
    except getopt.GetoptError:
        print('python MD_features.py -p <input PDB file path> -t <input trajectory file path> -i <input interaction file path> -o <output file name>')
        sys.exit(2)
    
    # Loop through parsed options and assign to respective variables
    for opt, arg in opts:
        if opt == '-h':
            print('python MD_features.py -p <input PDB file path> -t <input trajectory file path> -i <input interaction file path> -o <output file path>')
            sys.exit()
        elif opt in ("-p"):
            pdb_file = arg
        elif opt in ("-t"):
            traj_file = arg
        elif opt in ("-i"):
            inter_file = arg
        elif opt in ("-o"):
            res_feat_path = f'{arg}_res_feature.csv'
            pair_feat_path = f'{arg}_pair_feature.npz'

    # Load the trajectory and PDB file, and compute the required features
    traj, ca_atoms, N, n_frames = get_traj_info(traj_file, pdb_file)
    res_feat, cor_n = get_dssp_sasa_angle_rmsf_cor(traj, ca_atoms, N, n_frames)
    cont_map = get_contact_maps(inter_file, N, n_frames)

    # Concatenate pairwise contact map with correlation map
    pair_feat = np.concatenate([cont_map, cor_n.reshape(N, N, 1)], axis=2)

    # Save residue-level features as a CSV file and pairwise features as a compressed NumPy file
    res_feat.to_csv(res_feat_path, index=False)
    np.savez_compressed(pair_feat_path, array=pair_feat)

    print(f"Finished processing: {pdb_file}")

# Entry point of the script
if __name__ == "__main__":
    main(sys.argv[1:])
