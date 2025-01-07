# basic image
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# workplace
WORKDIR /RAPiDock
COPY . /RAPiDock

RUN apt-get update && apt-get upgrade -y && apt-get install -y wget bzip2 pkg-config build-essential python3-dev python3-pip libatlas-base-dev gfortran libfreetype6-dev

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
	bash Miniconda3-latest-Linux-x86_64.sh -b -f -p /opt/miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/opt/miniconda/bin:$PATH

RUN conda init bash

RUN conda env create -f rapidock_env.yaml -n RAPiDock && echo "conda activate RAPiDock" >> ~/.bashrc

ENV PATH=/opt/miniconda/envs/RAPiDock/bin:$PATH

RUN /opt/miniconda/envs/RAPiDock/bin/pip install --no-cache-dir -r requirement.txt

RUN /opt/miniconda/envs/RAPiDock/bin/python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'

RUN /opt/miniconda/envs/RAPiDock/bin/python inference.py --config default_inference_args.yaml --protein_peptide_csv data/protein_peptide_example.csv --output_dir results/default
