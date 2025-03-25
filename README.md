# SeqDance: A Protein Language Model for Representing Protein Dynamic Properties


## Abstract
Proteins function by folding amino acid sequences into dynamic structural ensembles. Despite the central role of protein dynamics, their complexity and the absence of efficient representation methods have hindered their incorporation into studies of protein function and mutation fitness, particularly in deep learning applications. To address this challenge, we present SeqDance, a protein language model designed to learn representations of protein dynamic properties directly from sequence. SeqDance was pre-trained on dynamic biophysical properties derived from over 30,400 molecular dynamics trajectories and 28,600 normal mode analyses. Our results demonstrate that SeqDance effectively captures local dynamic interactions, co-movement patterns, and global conformational features, even for proteins without homologs in the pre-training set. Furthermore, SeqDance improves predictions of protein fitness landscapes, disorder-to-order transition binding regions, and phase-separating proteins. By learning dynamic properties from sequence, SeqDance complements conventional evolution- and static structure-based methods, providing novel insights into protein behavior and function.


![SeqDance Pre-training Diagram](image/SeqDance_pretraining.png "Diagram of SeqDance Pre-training")


## SeqDance Pre-training and Usage
### Pre-training
SeqDance was trained using Python, PyTorch, and the Transformers library. For detailed environment setup, please refer to [SeqDance_env.yml](SeqDance_env.yml). For details on the model architecture and pre-training process, please refer to codes in the [model](./model/) directory.
```
conda env create -f SeqDance_env.yml
conda activate seqdance
cd model
torchrun --nnodes=1 --nproc_per_node=6 train_ddp.py
```
SeqDance is trained via [distributed data parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html). The detailed hyperparameters are listed in [config](./model/config.py). The pre-training took ten days on a server with six A6000 GPUs. 

We provide the training sequences and extracted features in [Hugging face](https://huggingface.co/datasets/ChaoHou/protein_dynamic_properties).


### Pre-trained weight
You can download the pre-trained SeqDance weights here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13909695.svg)](https://doi.org/10.5281/zenodo.13909695). Follow the instructions in [notebook/pretrained_seqdance_attention_embedding.ipynb](notebook/pretrained_seqdance_attention_embedding.ipynb) for how to extract pairwise features-related attentions and how to get residue level embeddings. Please note that this demo may take a few minutes to complete.


## Protein Dynamic Dataset
All pre-training datasets used in SeqDance are publicly available. 


| Source         | Description                                      | Number  | Method                            |
|----------------|--------------------------------------------------|---------|------------------------------------|
| [ATLAS](https://www.dsimb.inserm.fr/ATLAS/index.html)  | Ordered structures in PDB (no membrane proteins) | 1,516   | All-atom MD, 3x100 ns              |
| [PED](https://proteinensemble.org/)              | Disordered regions                             | 382     | Experimental and other methods     |
| [GPCRmd](https://www.gpcrmd.org/)               | Membrane proteins                              | 509     | All-atom MD, 3x500 ns              |
| [IDRome](https://github.com/KULL-Centre/_2023_Tesei_IDRome)       | Disordered regions                             | 28,058  | Coarse-grained MD, converted to all-atom |
| [ProteinFlow](https://github.com/adaptyvbio/ProteinFlow)          | Ordered structures in PDB                      | 28,631  | Normal mode analysis               |


### Coarse-grained MD Conversion
IDRome trajectories were converted to all-atom trajectories using [cg2all](https://github.com/huhlim/cg2all), with the following command:  
```
convert_cg2all -p top_ca.pdb -d traj.xtc -o traj_all.dcd -opdb top_all.pdb --cg CalphaBasedModel
```

## Feature Extraction
### MD Trajectory Feature Extraction
We extracted residue-level and pairwise dynamic features from MD trajectories:

- **Residue-level features**: Root mean square fluctuation (RMSF), surface area, secondary structure (eight classes), and dihedral angles (phi, psi, chi1).
- **Pairwise features**: Correlation of Cα movements, and frequencies of hydrogen bonds, salt bridges, Pi-cation, Pi-stacking, T-stacking, hydrophobic, and van der Waals interactions.

[GetContacts](https://getcontacts.github.io/) was used to extract nine types of interactions from MD trajectories:

```
cd data_prepare/molecular_dynamics
get_dynamic_contacts.py --itypes hb sb pc ps ts hp vdw --cores 2 --topology 3tvj_I.pdb --trajectory 3tvj_I_10frames.dcd --output 3tvj_I_10frames_contact.tsv
```


After extract interactions, you can use [MDTraj v1.9.9](https://www.mdtraj.org/) to generate the residue-level and pairwise features with:
```
cd data_prepare/molecular_dynamics
python MD_features.py -p 3tvj_I.pdb -t 3tvj_I_10frames.dcd -i 3tvj_I_10frames_contact.tsv -o 3tvj_I
```
`-p`: PDB structure file; `-t`: MD trajectory file (.dcd format here); `-i`: interaction tsv file from GetContacts; `-o`: file name for residue features and pairwise features.


### Normal Mode Analysis Feature Extraction
For NMA data, we used [ProDy v2.4.0](http://www.bahargroup.org/prody/index.html) to conduct the analysis. Normal modes were categorized into three frequency-based clusters. For each cluster, residue fluctuation and pairwise correlation maps were computed.  
```
cd data_prepare/normal_mode_analysis
python NMA_features.py -i 2g3r.pdb -o nma_residue_pair_features_2g3r
```
`-i`: PDB structure file; `-o`: file name for NMA features.

We recommend installing [GetContacts](https://getcontacts.github.io/), [MDTraj](https://www.mdtraj.org/), and [ProDy](http://www.bahargroup.org/prody/index.html) in different conda environments from the [SeqDance pre-training environment](SeqDance_env.yml). Installing all required packages took about a hour in our server.  
The feature extraction process is the most complicated step in our work, it took us over a month.