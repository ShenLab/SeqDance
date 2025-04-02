# SeqDance and ESMDance: Protein Language Models Trained on Protein Dynamic Properties


## Abstract
Proteins function by folding amino acid sequences into dynamic structural ensembles. Despite the important role of protein dynamics, their inherent complexity and the absence of efficient integration methods have limited their incorporation into deep learning models for studying protein behaviors and mutation effects. To address this, we developed SeqDance and ESMDance, protein language models pre-trained on dynamic biophysical properties derived from molecular dynamics trajectories of over 35,800 proteins and normal mode analyses of over 28,500 proteins. SeqDance, which operates solely on sequence input, captures local dynamic interactions and global conformation properties for both ordered and disordered proteins, even for proteins without homologs in the pre-training dataset. SeqDance predicted dynamic property changes are predictive of mutation effects on protein folding stability. ESMDance, which utilizes ESM2 outputs, significantly outperforms ESM2 in zero-shot prediction of mutation effects for designed and viral proteins. SeqDance and ESMDance offer novel insights into protein behaviors and mutation effects through the perspective of protein dynamics.

![SeqDance Pre-training Diagram](image/SeqDance_pretraining.png "Diagram of SeqDance Pre-training")

## !!! Data and weight
Training sequences and extracted features: [Hugging face](https://huggingface.co/datasets/ChaoHou/protein_dynamic_properties)  
Pre-trained SeqDance/ESMDance weights: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15047777.svg)](https://doi.org/10.5281/zenodo.15047777) (use Version v2), [Hugging face SeqDance](https://huggingface.co/ChaoHou/SeqDance), [Hugging face ESMDance](https://huggingface.co/ChaoHou/ESMDance).


## SeqDance/ESMDance Pre-training
SeqDance and ESMDance, both consist of Transformer encoders and dynamic property prediction heads. The Transformer encoder follows the same architecture as ESM2-35M, with 12 layers and 20 attention heads per layer. Both models take protein sequences as input and predict residue-level and pairwise dynamic properties. The dynamic property prediction heads containing 1.2 million trainable parameters.  

In SeqDance, all parameters were randomly initialized, allowing the model to learn dynamic properties from scratch. In ESMDance, all ESM2-35M parameters were frozen, enabling the model to leverage the evolutionary information captured by ESM2-35M to predict dynamic properties. For details on the model architecture and pre-training process, please refer to codes in the [model](./model/) directory.

#### Step 1
Before pre-training, please download and process the data as described in [Hugging face](https://huggingface.co/datasets/ChaoHou/protein_dynamic_properties) (Merging HDF5 Files). Change the file pathes in [config.py](./model/config.py).

#### Step 2
For detailed environment setup, please refer to [SeqDance_env.yml](SeqDance_env.yml). In our experiment, a new conda environment with pytorch=2.5.1, transformers=4.48.2, and h5py installed with conda also worked for the pre-training. 
```
conda env create -f SeqDance_env.yml
conda activate seqdance
python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 model/train_ddp.py
```

SeqDance/ESMDance were trained via [distributed data parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html). The detailed hyperparameters are listed in [config](./model/config.py). The pre-training took ten days on a server with four L40s GPUs. 


## SeqDance/ESMDance Usage
### Zero-shot prediction of mutation effect
Using SeqDance/ESMDance to predict the dynamic properties of both wild-type and mutated sequences, calculate the relative changes of dynamic properties after mutation, infer mutation effects with these relative changes.  
check the code [here](./notebook/zero_shot_mutation.ipynb) for how to load model, how to predict dynamic properties, and how to perform zero-shot prediction of mutation effects (ESMDance and ESM2)

![Zero-shot](image/zero_shot.png "Zero-shot")


### Application of embeddings


### SeqDance's attention capture protein dynamic interactions


## Protein Dynamic Dataset
All pre-training datasets used in SeqDance are publicly available. 


| Source         | Description                                      | Number  | Method                            |
|----------------|--------------------------------------------------|---------|------------------------------------|
| [mdCATH](https://huggingface.co/datasets/compsciencelab/mdCATH)  | Ordered structures in PDB | 5,392   | All-atom MD, 5x464 ns              |
| [ATLAS](https://www.dsimb.inserm.fr/ATLAS/index.html)  | Ordered structures in PDB (no membrane proteins) | 1,516   | All-atom MD, 3x100 ns              |
| [PED](https://proteinensemble.org/)              | Disordered regions                             | 382     | Experimental and other methods     |
| [GPCRmd](https://www.gpcrmd.org/)               | Membrane proteins                              | 509     | All-atom MD, 3x500 ns              |
| [IDRome](https://github.com/KULL-Centre/_2023_Tesei_IDRome)       | Disordered regions                             | 28,058  | Coarse-grained MD, converted to all-atom |
| [ProteinFlow](https://github.com/adaptyvbio/ProteinFlow)          | Ordered structures in PDB                      | 28,546  | Normal mode analysis               |


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


## Citation
SeqDance: A Protein Language Model for Representing Protein Dynamic Properties
Chao Hou, Yufeng Shen
bioRxiv 2024.10.11.617911; doi: https://doi.org/10.1101/2024.10.11.617911
