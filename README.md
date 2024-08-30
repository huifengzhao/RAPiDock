# RAPiDock: Pushing the Boundaries of Protein-peptide Docking with Rational and Accurate Diffusion Generative Model

------

![workflow](./figures/workflow.jpg)

## Table of Contents:

------

- [Description](#description)
- [Software prerequisites](#software-prerequisites)
- [Setup Environment](#setup-environment)
  - [Clone the current repo](#Clone-the-current-repo)
  - [Install option 1: Install via conda .yaml file](#Install-option-1:-Install-via-conda-.yaml-file)
  - [Install option 2: Install manually](#Install-option-2:-Install-manually)
- [Protein-peptide Docking](#Protein-peptide-docking)
  - [Input formats](#Input formats)
  - [Supported residues](#supported residues)
  - [Docking prediction](#docking prediction)
  - [Visualization](#Visualization)

## Description

-------------

RAPiDock is a diffusion generative model designed for rational, accurate, and rapid protein-peptide docking at an all-atomic level.

## Software prerequisites

--------------------

RAPiDock relies on external software/libraries to handle protein and peptide dataset files, to compute atom features, and to perform neural network calculations. The following is the list of required libraries and programs, as well as the version on which it was testes.

- [ ] [Python](https://www.python.org/) (3.9).
- [ ] [Pytorch](https://pytorch.org/) (1.11.0). Use to model, train, and evaluate the actual neural networks.
- [ ] [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (11.5.1). 
- [ ] [PyG](https://www.pyg.org/) (2.1.0). Used for implementing neural networks.
- [ ] [MDAnalysis](https://www.mdanalysis.org/) (2.6.1). To handle residue acids related data.
- [ ] [BioPython](https://github.com/biopython/biopython) (1.84). To parse PDB files.
- [ ] [E3NN](https://e3nn.org/) (0.5.1). Used to implement E(3) equivariant neural network.
- [ ] [RDKit](https://www.rdkit.org/) (2022.9.5). Used to process and extract atomic features.
- [ ] *[PyRosetta](https://www.pyrosetta.org/) (2024.10). Used to further optimize the peptides generated by the model and rescore the generated peptide. (**Optional**)

## Setup Environment

-------------

We will set up the environment using [Anaconda](https://docs.anaconda.com/anaconda/install/index.html).

### Clone the current repo

```shell
git clone https://github.com/huifengzhao/RAPiDock.git
```

### Install option 1: Install via conda .yaml file

We can easy install the environment by using the provided ***rapidock_env.yaml*** and ***requirement.txt*** files.

```shell
conda env create -f rapidock_env.yaml -n RAPiDock
conda activate RAPiDock # activating the constructed environment
pip install -r requirement.txt # We separated the dependencies of conda and pip for a better experience of environment installation.
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()' # Installation of PyRosetta (Optional)
```

### Install option 2: Install manually

If we fail to install the environment via provided `.yaml` file, we can also install the environment manually through the following steps:

```shell
conda create -n RAPiDock cudatoolkit=11.5.1 pytorch=1.11.0 pyg=2.1.0 mkl=2023.1.0 python=3.9 numpy pyyaml -c conda-forge -c pytorch -c pyg
pip install MDAnalysis==2.6.1 pandas==2.1.0 e3nn==0.5.1 rdkit-pypi==2022.9.5 fair-esm==2.0.0 pyrosetta-installer==0.1.0
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()' # Installation of PyRosetta (Optional)
```

## Protein-peptide Docking

----------------------

Now, we can run the code as described below.

### Input formats

We support multiple input formats depending on specific tasks.

- **Local docking**: When we try to perform local docking, the binding pocket and/or the protein structure are always known.  So, we need to prepare:

  - **Protein**: The `.pdb` file of the protein or its pocket (for better accuracy). 

    > [!TIP]
    >
    > The pocket structure of protein can be generated by using `pocket_trunction.py`.

  - **Peptide**: The `.pdb` file or sequence of the peptide. 

    > [!NOTE]
    >
    > `.pdb` file of peptide is only used for sequence extracting, and no 3D information of peptide will be kept.

  For example:

  ```shell
  python inference.py [--ohter options] --protein_description protein.pdb --peptide_description peptide.pdb
  ```

  or just using the sequence information:

  ```shell
  python inference.py [--ohter options] --protein_description protein.pdb --peptide_description HKILHRLLQDS
  ```

- **Global docking**: When we try to perform global docking, we have no idea about the binding pocket of the protein, and we even have no idea about the protein structure. So, we need to prepare:

  - **Protein**: The `.pdb` file of the protein or sequence of the protein (We can use the default ESMFold method to fold the protein). 

    > [!TIP]
    >
    > We can also use other ways, such as using [AlphaFold](https://golgi.sandbox.google.com/) or homology modeling to generate the protein structure.

  - **Peptide**: The `.pdb` file or sequence of the peptide. 

  For example:

  ```shell
  python inference.py [--ohter options] --protein_description ...SLAPYASLTEIEHLVQSVCKSYRETCQLRLEDLLRQRSNIFSREEVTGYQ... --HKILHRLLQDS
  ```

- **Virtual Screening**: When we try to perform virtual screening, we always have the information about the binding pocket and/or the protein structure, and a multiple sequence of peptides. So, we provide Multi-task submission mode:

  - **Protein and peptide description**: A `.csv` file including  multiple protein-peptide pair information. The `.csv` file looks like:

    | complex_name | protein_description  | peptide_description |
    | :----------: | :------------------: | :-----------------: |
    |   complex1   | /path/to/protein.pdb |     HKILHRLLQDS     |
    |   complex2   | /path/to/protein.pdb |    EKHKILHRLLQDS    |
    |     ...      |         ...          |         ...         |
    |   complexN   | /path/to/protein.pdb |      LSGFMELCQ      |

  Then, we can simply launch the model by using the following command:

  ```shell
  python inference.py [--ohter options] --protein_peptide_csv /path/to/virtual_screening.csv
  ```

- **Customized Multi-tasks**: We can also perform multiple customized tasks in one shot, using a `.csv` file. 

  - **Protein and peptide description**: A `.csv` file including  multiple protein-peptide pair information. The `.csv` file looks like:
    | complex_name |                   protein_description                    | peptide_description  |
    | :----------: | :------------------------------------------------------: | :------------------: |
    |   complex1   |                   /path/to/protein.pdb                   |      AAAARLLQDS      |
    |   complex2   | ...SLAPYASLTEIEHLVQSVCKSYRETCQLRLEDLLRQRSNIFSREEVTGYQ... | /path/to/peptide.pdb |
    |     ...      |                           ...                            |         ...          |
    |   complexN   |                   /path/to/protein.pdb                   | /path/to/peptide.pdb |

  Then, we can simply launch the model by using the following command:

  ```shell
  python inference.py [--ohter options] --protein_peptide_csv /path/to/customized_tasks.csv
  ```

### Supported residues

In current version of RAPiDock, we support 92 types of residues for protein-peptide binding pattern prediction. The supported residues are illustrated bellow:

![supported_residues](./figures/supported_residues.jpg)

For the use of 92 types of residues, we define a special format for model input:

|   G   |   A   |   V   |   I   |   L   |   M   |   F   |   Y   |   W   |   P   |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   S   |   T   |   N   |   Q   |   D   |   E   |   C   |   R   |   H   |   K   |
| [HYP] | [SEP] | [TYS] | [ALY] | [TPO] | [PTR] | [DAL] | [MLE] | [M3L] | [DLE] |
| [DLY] | [AIB] | [MSE] | [DPR] | [MVA] | [NLE] | [MLY] | [SAR] | [ABA] | [FME] |
| [DAR] | [ORN] | [CGU] | [DPN] | [DTY] | [DTR] | [4BF] | [DGL] | [DCY] | [MK8] |
| [MP8] | [GHP] | [ALC] | [BMT] | [MLZ] | [DVA] | [3FG] | [DAS] | [7ID] | [DSN] |
| [AR7] | [MEA] | [PHI] | [MAA] | [LPD] | [KCR] | [PCA] | [DGN] | [2MR] | [DHI] |
| [ASA] | [MLU] | [YCP] | [DSG] | [DTH] | [OMY] | [FP9] | [DPP] | [HCS] | [SET] |
| [DBB] | [BTK] | [DAM] | [IIL] | [3MY] | [SLL] | [PFF] | [HRG] | [DIL] | [DNE] |
| [MED] | [D0C] |       |       |       |       |       |       |       |       |

Then, we can simply launch the model for predicting peptide with non-canonical amino acids by using the following command:

```shell
python inference.py [--ohter options] --protein_description protein.pdb --peptide_description HK[HYP]RL[PTR]QDS
```

### Docking prediction

Then, we are ready to run inference:

```shell
python inference.py --config default_inference_args.yaml --protein_peptide_csv data/protein_peptide_example.csv --output_dir results/default
```

By default, we will use 5 CPUs for computing, if we want to change the number of CPUs, we can simply:

```shell
python inference.py --config default_inference_args.yaml --protein_peptide_csv data/protein_peptide_example.csv --output_dir results/default --cpu 10
```

### Visualization

We also provide visualization of the model inference process. We can simply:

```
python inference.py --config default_inference_args.yaml --protein_peptide_csv data/protein_peptide_example.csv --output_dir results/default --save_visualisation
```

Then the inference process of generated peptide will be save by name of `rankN_reverseprocess.pdb`

hope you enjoy the processing of using RAPiDock.
