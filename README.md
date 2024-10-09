# SeqDance: A Protein Language Model for Representing Protein Dynamic Properties

## Abstract
Proteins function by folding amino acid sequences into dynamic structural ensembles. Despite the central role of protein dynamics, their complexity and the absence of efficient representation methods have hindered their incorporation into studies of protein function and mutation fitness, particularly in deep learning applications. To address this challenge, we present SeqDance, a protein language model designed to learn representations of protein dynamic properties directly from sequence. SeqDance is pre-trained on dynamic biophysical properties derived from over 30,400 molecular dynamics trajectories and 28,600 normal mode analyses. Our results demonstrate that SeqDance effectively captures local dynamic interactions, co-movement patterns, and global conformational features, even for proteins without homologs in the pre-training set. Furthermore, SeqDance improves predictions of protein fitness landscapes, disorder-to-order transition binding regions, and phase-separating proteins. By learning dynamic properties from sequence, SeqDance complements traditional evolution- and structure-based methods, providing novel insights into protein behavior and function.

![SeqDance Pre-training Diagram](image/SeqDance_pretraining.png "Diagram of SeqDance Pre-training")

## Environment
SeqDance was trained using Python (v3.12.2), PyTorch (v2.2.0), and the Transformers library (v4.39.1). For detailed environment setup, please refer to [SeqDance_env.yml](SeqDance_env.yml).

## Protein Dynamic Dataset
All pre-training datasets used in SeqDance are publicly available:

| Source         | Description                                      | Number  | Method                            |
|----------------|--------------------------------------------------|---------|------------------------------------|
| **High resolution** |                                              |         |                                    |
| [ATLAS](https://www.dsimb.inserm.fr/ATLAS/index.html)  | Ordered structures in PDB (no membrane proteins) | 1,516   | All-atom MD, 3x100 ns              |
| [PED](https://proteinensemble.org/)              | Disordered regions                             | 382     | Experimental and other methods     |
| [GPCRmd](https://www.gpcrmd.org/)               | Membrane proteins                              | 509     | All-atom MD, 3x500 ns              |
| **Low resolution**  |                                              |         |                                    |
| [IDRome](https://github.com/KULL-Centre/_2023_Tesei_IDRome)       | Disordered regions                             | 28,058  | Coarse-grained MD, converted to all-atom |
| [ProteinFlow](https://github.com/adaptyvbio/ProteinFlow)          | Ordered structures in PDB                      | 28,631  | Normal mode analysis               |

For details on how we extracted features from molecular dynamics (MD) trajectories and normal mode analysis (NMA), check the code in [data_prepare](./data_prepare/). We extracted residue-level and pairwise dynamic features, capturing structural ensemble distributions:

- **Residue-level features**: Root mean square fluctuation (RMSF), surface area, secondary structure (eight classes), and dihedral angles (phi, psi, chi1).
- **Pairwise features**: Correlation of Cα movements, and frequencies of hydrogen bonds, salt bridges, Pi-cation, Pi-stacking, T-stacking, hydrophobic, and van der Waals interactions.

For NMA data, we categorized normal modes into three frequency-based clusters. For each cluster, residue fluctuation and pairwise correlation maps were computed.

If you are interested in using the extracted features, please contact us.
