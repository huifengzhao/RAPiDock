##########################################################################
# File Name: peptide_feature.py
# Author: huifeng
# mail: huifengzhao@zju.edu.cn
# Created Time: Thu 19 Oct 2023 09:31:15 AM CST
#########################################################################

import math
import numpy as np
import torch
from torch_scatter import scatter_max,scatter_min
from utils.PeptideBuilder import get_edges_from_sequence
import torch.nn.functional as F
import torch_cluster
from MDAnalysis.analysis import distances
from utils.dataset_utils import read_pdb_with_seq, three_to_one, standard_residue_sort
from io import StringIO
import MDAnalysis
from rdkit.Chem import AllChem, RemoveHs
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
import os

three2idx = {k:v for v, k in enumerate(['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 
										'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 
										'GLU', 'LYS', 'ARG', 'HIS', 'UNK', 'HYP', 'SEP', 'TYS', 'ALY', 'TPO', 'PTR', 'DAL', 'MLE', 'M3L', 'DLE', 'DLY', 'AIB', 'MSE', 'DPR', 'MVA', 'NLE', 'MLY', 'SAR', 'ABA', 'DAR', 'ORN', 'CGU', 'DPN', 'DTY', 'DTR', '4BF', 'DGL', 'DCY', 'MK8', 'MP8', 'GHP', 'ALC', 'BMT', 'MLZ', 'ASJ', 'DVA', '3FG', 'DAS', '7ID', 'DSN', 'AR7', 'MEA', 'FGA', 'PHI', 'MAA', 'LPD', 'KCR', 'B3L', 'PSA', 'PCA', 'DGN', '2MR', 'DHI', 'XPC', 'ASA', 'MLU', 'YCP', 'BIL', 'DSG', 'DTH', 'OMY', 'ACB', 'B3A', 'FP9', 'DPP', 'HCS', 'SET', 'DBB', 'BTK', 'DAM', 'IIL', 'B3K', '3MY', 'SLL', 'PFF', 'B3D', 'HRG', "DIL", "MED", "D0C", "DNE", "FME", 'X'])}
three2self = {v:v for v in ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 
										'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 
										'GLU', 'LYS', 'ARG', 'HIS', 'UNK', 'HYP', 'SEP', 'TYS', 'ALY', 'TPO', 'PTR', 'DAL', 'MLE', 'M3L', 'DLE', 'DLY', 'AIB', 'MSE', 'DPR', 'MVA', 'NLE', 'MLY', 'SAR', 'ABA', 'DAR', 'ORN', 'CGU', 'DPN', 'DTY', 'DTR', '4BF', 'DGL', 'DCY', 'MK8', 'MP8', 'GHP', 'ALC', 'BMT', 'MLZ', 'ASJ', 'DVA', '3FG', 'DAS', '7ID', 'DSN', 'AR7', 'MEA', 'FGA', 'PHI', 'MAA', 'LPD', 'KCR', 'B3L', 'PSA', 'PCA', 'DGN', '2MR', 'DHI', 'XPC', 'ASA', 'MLU', 'YCP', 'BIL', 'DSG', 'DTH', 'OMY', 'ACB', 'B3A', 'FP9', 'DPP', 'HCS', 'SET', 'DBB', 'BTK', 'DAM', 'IIL', 'B3K', '3MY', 'SLL', 'PFF', 'B3D', 'HRG', "DIL", "MED", "D0C", "DNE", "FME"]}

aa2tip = [
        "CA", # gly
        "CB", # ala
        "CB", # val
        "CG", # leu
        "CD1", # ile
        "CG", # pro
        "CZ", # phe
        "OH", # tyr
        "CH2", # trp
        "OG", # ser
        "OG1", # thr
        "SG", # cys
        "SD", # met
        "ND2", # asn
        "NE2", # gln
        "CG", # asp
        "CD", # glu
        "NZ", # lys
        "CZ", # arg
        "NE2", # his
        "CB", # unknown (gap etc)
        "OD1", # hyp
        "P", # sep
        "S", # tys
        "OH", # aly
        "P", # tpo
        "P", # ptr
        "CB", # dal
        "CG", # mle
        "NZ", # m3l
        "CG", # dle
        "NZ", # dly
        "CA", # aib
        "SE", # mse
        "CG", # dpr
        "CB", # mva
        "CE", # nle
        "NZ", # mly
        "CA", # sar
        "CG", # aba
        "CZ", # dar
        "NE", # orn
        "CG", # cgu
        "CZ", # dpn
        "OH", # dty
        "CH2", # dtr
        "BR", # 4bf
        "CD", # dgl
        "SG", # dcy
        "CE", # mk8
        "CE", # mp8
        "O4", # ghp
        "CZ", # alc
        "CH", # bmt
        "NZ", # mlz
        "CB", # asj
        "CB", # dva
        "CZ", # 3fg
        "CG", # das
        "CZ2", # 7id
        "OG", # dsn
        "CZ", # ar7
        "CZ", # mea
        "CB", # fga
        "I", # phi
        "CB", # maa
        "CG", # lpd
        "CH3", # kcr
        "CB", # b3l
        "CB", # psa
        "OE", # pca
        "CD", # dgn
        "CZ", # 2mr
        "NE2", # dhi
        "CB", # xpc
        "CG", # asa
        "CG", # mlu
        "CG", # ycp
        "CB", # bil
        "CG", # dsg
        "CB", # dth
        "OCZ", # omy
        "CB", # acb
        "CB", # b3a
        "FD", # fp9
        "NG", # dpp
        "SD", # hcs
        "OG", # set
        "CG", # dbb
        "CAA", # btk
        "CB", # dam
        "CD1", # iil
        "CB", # b3k
        "OBD", # 3my
        "CP", # sll
        "F", # pff
        "SB", # b3d
        "CZ", # hrg
        "CD1", # dil
        "SD", # med
        "CL", # d0c
        "CE", # dne
        "SD", # fme
        "CB", # x
        ]

atomname2idx=[
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'OXT', 'X'])},# GLY
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'OXT', 'X'])},# ALA
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'OXT', 'X'])},# VAL
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'OXT', 'X'])},# LEU
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', 'OXT', 'X'])},# ILE
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OXT', 'X'])},# PRO
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OXT', 'X'])},# PHE
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', 'OXT', 'X'])},# TYR
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'OXT', 'X'])},# TRP
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'OG', 'OXT', 'X'])},# SER
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', 'OXT', 'X'])},# THR
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'SG', 'OXT', 'X'])},# CYS
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', 'OXT', 'X'])},# MET
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2', 'OXT', 'X'])},# ASN
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2', 'OXT', 'X'])},# GLN
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2', 'OXT', 'X'])},# ASP
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', 'OXT', 'X'])},# GLU
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', 'OXT', 'X'])},# LYS
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', 'OXT', 'X'])},# ARG
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2', 'OXT', 'X'])},# HIS
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'OG', 'OXT', 'X'])},# UNK
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OD1', 'OXT', 'X'])},# HYP
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'OG', 'P', 'O1P', 'O2P', 'O3P', 'OXT', 'X'])},# SEP
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', 'S', 'O1', 'O2', 'O3', 'OXT', 'X'])},# TYS
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'OH', 'CH', 'CH3', 'NZ', 'CE', 'CD', 'CG', 'CB', 'OXT', 'X'])},# ALY
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG2', 'OG1', 'P', 'O1P', 'O2P', 'O3P', 'OXT', 'X'])},# TPO
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', 'P', 'O1P', 'O2P', 'O3P', 'OXT', 'X'])},# PTR
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'OXT', 'X'])},# DAL
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CN', 'OXT', 'X'])},# MLE
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', 'CM1', 'CM2', 'CM3', 'OXT', 'X'])},# M3L
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'OXT', 'X'])},# DLE
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', 'OXT', 'X'])},# DLY
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB1', 'CB2', 'OXT', 'X'])},# AIB
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'SE', 'CE', 'OXT', 'X'])},# MSE
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OXT', 'X'])},# DPR
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CN', 'OXT', 'X'])},# MVA
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'OXT', 'X'])},# NLE
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', 'CH1', 'CH2', 'OXT', 'X'])},# MLY
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CN', 'OXT', 'X'])},# SAR
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'OXT', 'X'])},# ABA
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', 'OXT', 'X'])},# DAR
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'OXT', 'X'])},# ORN
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'OE11', 'OE12', 'OE21', 'OE22', 'OXT', 'X'])},# CGU
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OXT', 'X'])},# DPN
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', 'OXT', 'X'])},# DTY
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'NE1', 'CE2', 'CZ2', 'CH2', 'CZ3', 'CE3', 'CD2', 'OXT', 'X'])},# DTR
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CD1', 'CE1', 'CZ', 'BR', 'CE2', 'CD2', 'CG', 'CB', 'OXT', 'X'])},# 4BF
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', 'OXT', 'X'])},# DGL
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'SG', 'OXT', 'X'])},# DCY
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CD', 'CE', 'CG', 'CB1', 'OXT', 'X'])},# MK8
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CD', 'CE', 'CG', 'OXT', 'X'])},# MP8
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'C1', 'C2', 'C3', 'C4', 'O4', 'C5', 'C6', 'OXT', 'X'])},# GHP
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OXT', 'X'])},# ALC
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', 'CD1', 'CD2', 'CE', 'CZ', 'CH', 'CN', 'OXT', 'X'])},# BMT
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', 'CM', 'OXT', 'X'])},# MLZ
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2', 'OXT', 'X'])},# ASJ not alpha amino
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'OXT', 'X'])},# DVA
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'OD1', 'CD1', 'CG1', 'CZ', 'CD2', 'OD2', 'CG2', 'CB', 'OXT', 'X'])},# 3FG
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2', 'OXT', 'X'])},# DAS
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'N2', 'CA2', 'CO2', 'O2', 'OX2', 'CB2', 'CG2', 'CD2', 'NE2', 'CZ2', 'NH1', 'NH2', 'OD1', 'OXT', 'X'])},# 7ID
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'OG', 'OXT', 'X'])},# DSN
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', 'OXT', 'X'])},# AR7
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'C1', 'CB', 'CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2', 'OXT', 'X'])},# MEA
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OXT', 'X'])},# FGA not alpha amino
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'I', 'OXT', 'X'])},# PHI
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CM', 'OXT', 'X'])},# MAA
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CD', 'CG', 'CB', 'N2', 'OXT', 'X'])},# LPD
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', 'CH', 'OH', 'CX', 'CY', 'CH3', 'OXT', 'X'])},# KCR
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE2', 'CE1', 'OXT', 'X'])},# B3L not alpha amino
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'CH', 'OH', 'CM', 'OXT', 'X'])},# PSA not alpha amino
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE', 'OXT', 'X'])},# PCA
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2', 'OXT', 'X'])},# DGN
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'CQ1', 'NH2', 'CQ2', 'OXT', 'X'])},# 2MR
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2', 'OXT', 'X'])},# DHI
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND', 'CE', 'OXT', 'X'])},# XPC not alpha amino
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2', 'OXT', 'X'])},# ASA
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CN', 'OXT', 'X'])},# MLU
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CD', 'CE', 'CG', 'OXT', 'X'])},# YCP
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD2', 'CD1', 'CE1', 'OXT', 'X'])},# BIL not alpha amino
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2', 'OXT', 'X'])},# DSG
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG2', 'OG1', 'OXT', 'X'])},# DTH
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'OCZ', 'CE2', 'CE1', 'CZ', 'CG', 'CD2', 'CD1', 'CB', 'CL', 'ODE', 'OXT', 'X'])},# OMY
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'C4', 'OXT', 'X'])},# ACB not alpha amino
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CG', 'CB', 'OXT', 'X'])},# B3A not alpha amino
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CD', 'FD', 'CG', 'OXT', 'X'])},# FP9
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'NG', 'OXT', 'X'])},# DPP
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'OXT', 'X'])},# HCS
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'OG', 'NT', 'OXT', 'X'])},# SET
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'OXT', 'X'])},# DBB
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CD', 'CE', 'CG', 'NZ', 'CAA', 'OAD', 'CAF', 'CAJ', 'CAN', 'OXT', 'X'])},# BTK
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CM', 'OXT', 'X'])},# DAM
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG2', 'CG1', 'CD1', 'OXT', 'X'])},# IIL
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'CF', 'NZ', 'OXT', 'X'])},# B3K not alpha amino
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'OBD', 'CZ', 'CE2', 'CD2', 'CL', 'CE1', 'CD1', 'CG', 'CB', 'OXT', 'X'])},# 3MY
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CD', 'CE', 'CG', 'CK', 'CL', 'CP', 'CX', 'OX', 'NZ', 'OP1', 'OP2', 'OXT', 'X'])},# SLL
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'F', 'OXT', 'X'])},# PFF
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'OE1', 'CD', 'OE2', 'CG', 'CB', 'OXT', 'X'])},# B3D not alpha amino
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', "CG'", 'CD', 'NE', 'CZ', 'NH1', 'NH2', 'OXT', 'X'])},# HRG
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', 'OXT', 'X'])},# DIL
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', 'OXT', 'X'])},# MED
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'CL', 'OXT', 'X'])},# D0C
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'OXT', 'X'])},# DNE
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', 'CN', 'O1', 'OXT', 'X'])},# FME
    {k:v for v, k in enumerate(['N', 'CA', 'C', 'O', 'CB', 'OXT', 'X'])},# X
]

aa2tip_idx = [atomname2idx[i].get(aa2tip[i],atomname2idx[i]['X']) for i in range(len(three2idx))]

allowable_features = {
    'possible_symbol_list': ["H", "B", "C", "N", "O", "F", "Mg", "Si", "P", "S", "Cl", "Cu", "Zn", "Se", "Br", "Sn", "I"] + ['misc'],
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_total_valence_list': [0, 1, 2, 3, 4, 5, 6, 7, 'misc'],
    'possible_is_aromatic_list': [False, True],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
    'possible_is_in_ring8_list': [False, True],
}

def safe_index(l, e):
    """ Return index of element e in list l. If e is not present, return the last index """
    try:
        return l.index(e)
    except:
        return len(l) - 1

def lig_atom_featurizer(pep_noh):
    # mol = read_mols('.',pep_noh, remove_hs=False,sanitize=True)
    mol = read_pdb_with_seq(pep_noh)
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        atom_features_list.append([
            safe_index(allowable_features['possible_symbol_list'], atom.GetSymbol()),
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            safe_index(allowable_features['possible_total_valence_list'],atom.GetTotalValence()),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            safe_index(allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx)),
            allowable_features['possible_is_in_ring3_list'].index(ringinfo.IsAtomInRingOfSize(idx, 3)),
            allowable_features['possible_is_in_ring4_list'].index(ringinfo.IsAtomInRingOfSize(idx, 4)),
            allowable_features['possible_is_in_ring5_list'].index(ringinfo.IsAtomInRingOfSize(idx, 5)),
            allowable_features['possible_is_in_ring6_list'].index(ringinfo.IsAtomInRingOfSize(idx, 6)),
            allowable_features['possible_is_in_ring7_list'].index(ringinfo.IsAtomInRingOfSize(idx, 7)),
            allowable_features['possible_is_in_ring8_list'].index(ringinfo.IsAtomInRingOfSize(idx, 8)),
        ])

    return torch.tensor(atom_features_list)

def get_chiralcenters(pep_noh):
    # mol = read_mols('.',pep_noh, remove_hs=False,sanitize=True)
    mol = read_pdb_with_seq(pep_noh)
    try:
        chiralcenters = Chem.FindMolChiralCenters(mol,force=True,includeUnassigned=True, useLegacyImplementation=False)
    except:
        chiralcenters = []
    chiral_arr = torch.zeros([mol.GetNumAtoms()])
    for (i, rs) in chiralcenters:
        if rs == 'R':
            chiral_arr[i] =1
        elif rs == 'S':
            chiral_arr[i] =2 
        else:
            chiral_arr[i] =3 
    return chiral_arr


def find_backbone_bonds(noh_pep):
    
    backbone_row = []
    backbone_col = []
    # 初始化前一个氨基酸
    prev_residue = None
    res_atoms_dic = {}
    # 遍历链中的氨基酸
    for residue in noh_pep.residues:
        # if residue.id[0] == " ": 只保留标准氨基酸
        # 如果不是第一个氨基酸，检查是否与前一个氨基酸形成酰胺键
        nitrogen = residue.atoms.select_atoms('name N')
        carbon_alpha = residue.atoms.select_atoms('name CA')
        carbon = residue.atoms.select_atoms('name C')
        oxy = residue.atoms.select_atoms('name O')
        backbone_row += [nitrogen.indices[0], carbon_alpha.indices[0]]
        backbone_row += [carbon_alpha.indices[0], carbon.indices[0]]
        backbone_row += [carbon.indices[0], oxy.indices[0]]
        backbone_col += [carbon_alpha.indices[0],nitrogen.indices[0]]
        backbone_col += [carbon.indices[0],carbon_alpha.indices[0]]
        backbone_col += [oxy.indices[0],carbon.indices[0]]
        res_atoms_dic[residue.ix]= residue.atoms.indices
        
        if prev_residue is not None:
            # 查找前一个氨基酸的碳原子和当前氨基酸的氮原子
            prev_carbon = prev_residue.atoms.select_atoms('name C')
            current_nitrogen = residue.atoms.select_atoms('name N')

            # 输出前一个氨基酸的名称、序号、碳原子和当前氨基酸的氮原子
            # print(f"Peptide Bond Between Residue {prev_residue.id[1]} ({prev_residue.get_resname()}) and Residue {residue.id[1]} ({residue.get_resname()})")
            # print(f"Carbon Atom ({prev_carbon.name}) Serial Number: {prev_carbon.get_serial_number()}")
            # print(f"Nitrogen Atom ({current_nitrogen.name}) Serial Number: {current_nitrogen.get_serial_number()}")
            backbone_row += [prev_carbon.indices[0], current_nitrogen.indices[0]]
            backbone_col += [current_nitrogen.indices[0],prev_carbon.indices[0]]
        # 更新前一个氨基酸
        prev_residue = residue

    backbone_edge_index = torch.tensor([backbone_row, backbone_col], dtype=torch.long)
    return backbone_edge_index,res_atoms_dic

def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        raise ValueError('Expect the format of the molecule_file to be '
                         'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))
    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn('Unable to compute charges for the molecule.')

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except Exception as e:
        print(e)
        print("RDKit was unable to read the molecule.")
        return None

    return mol

def read_mols(pdbbind_dir, name, remove_hs=False,sanitize=False):
    file = os.path.join(pdbbind_dir, name)
    lig = read_molecule(file, remove_hs=remove_hs, sanitize=sanitize)
    return lig

def find_sidechain_bonds(pep_noh,coords,backbone_edge_index):
    mol = read_mols('.',pep_noh, remove_hs=False)
    mol_maybe_noh = RemoveHs(mol, sanitize=True)
    assert np.sum(~(mol_maybe_noh.GetConformer().GetPositions().astype(np.float32) == np.asarray(coords))) == 0
    row, col = [], []
    for bond in mol_maybe_noh.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
    edge_index = torch.tensor([row, col], dtype=torch.long)
    sidechain_edge_index = edge_index.T[~torch.tensor([(i in backbone_edge_index.T.tolist()) for i in edge_index.T.tolist()])].T
    return edge_index,sidechain_edge_index
    
def obtain_dihediral_angles(res):
    angle_lis = [0, 0, 0, 0]
    for idx, angle in enumerate([res.phi_selection, res.psi_selection, res.omega_selection, res.chi1_selection]):
        try:
            angle_lis[idx] = angle().dihedral.value()
        except:
            continue
    return angle_lis

def calc_dist(res1, res2):
	dist_array = distances.distance_array(res1.atoms.positions,res2.atoms.positions)
	return dist_array

def obatin_edge(res_lis, src, dst):
    dist = calc_dist(res_lis[src], res_lis[dst])
    return dist.min()*0.1, dst.max()*0.1

def check_connect(res_lis, i, j):
    if abs(i-j) == 1 and res_lis[i].segid == res_lis[j].segid:
        return 1
    else:
        return 0

def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def get_orientations(X):
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def get_sidechains(n, ca, c):
    c, n = _normalize(c - ca), _normalize(n - ca)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec


def positional_embeddings_v1(edge_index,device='cpu',
                                num_embeddings=16,
                                period_range=[2, 1000]):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    d = edge_index[0] - edge_index[1]
    # raw
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32)
        * -(torch.log(torch.tensor(10000.0)) / num_embeddings)
    ).to(device)
    angles = d.unsqueeze(-1) * frequency
    # new
    max_relative_feature = 32
    d = torch.clip(d + max_relative_feature, 0, 2 * max_relative_feature)
    d_onehot = F.one_hot(d, 2 * max_relative_feature + 1)
    E = torch.cat((torch.cos(angles), torch.sin(angles), d_onehot), -1)
    return E

def get_ori_peptide_feature_mda(peptide_file, lm_embedding_chains=None, match = False, modify = None):    
    peptide_mol = None
    peptide_ori = MDAnalysis.Universe(peptide_file)
    
    if match:
        peptide_match = os.path.join(
            os.path.dirname(peptide_file), 'peptide_match.pdb'
        )
        peptide_mol = MDAnalysis.Universe(peptide_match)
    else:
        peptide_mol = MDAnalysis.Universe(peptide_file)
        
    seq_str = ''.join([three_to_one[res_name] if res_name in three_to_one.keys() else f'[{res_name}]' for res_name in peptide_mol.residues.resnames])
    oxt = len(peptide_mol.atoms.select_atoms('name OXT')) == 1
    all_edge_index = (get_edges_from_sequence(seq_str, oxt=oxt)[:,:2] -1)
    all_edge_index = np.concatenate([all_edge_index,all_edge_index[:,[1,0]]],axis=-1).reshape(-1,2)
    
    with torch.no_grad():
        coords = []
        ori_coords = []
        c_alpha_coords = []
        valid_chain_ids = []
        valid_lm_embeddings = []
        lengths = []
        pure_res_lis, seq = [], []

        for i,chain in enumerate(peptide_mol.segments):
            chain_coords = []  # num_residues, num_atoms, 3
            chain_c_alpha_coords = []
            count = 0
            chain_seq = []
            chain_pure_res_lis = []
            trans = {}
            _ = -1
            lm_embedding_chain = lm_embedding_chains[i] if lm_embedding_chains is not None else None
            for res_idx, residue in enumerate(chain.residues):
                residue_coords = None
                c_alpha, n, c, o = None, None, None, None
                res_name = residue.resname.strip()
                res_atoms = residue.atoms.select_atoms('not type H')
                c_alpha = res_atoms.select_atoms("name CA")
                n = res_atoms.select_atoms("name N")
                c = res_atoms.select_atoms("name C")
                o = res_atoms.select_atoms("name O")
                residue_coords = res_atoms.positions
                if len(c_alpha) == 1 and len(n) == 1 and len(c) == 1 and len(o) == 1:
                    # only append residue if it is an amino acid and not some weird molecule that is part of the complex
                    _ += 1
                    chain_c_alpha_coords.append(c_alpha.positions[0])
                    chain_coords.append(residue_coords)
                    
                    chain_seq.append(three2idx[three2self.get(res_name, 'X')])
                    chain_pure_res_lis.append(residue)
                    trans[int(residue.resid) if residue.icode == '' else f"{residue.resid}{residue.icode.strip()}"] = _
                    count += 1
                    
            # Normalize the order of residues
            real_idx = [trans[idx] for idx in sorted(set(trans.keys()),key=standard_residue_sort)]        
                
            chain_coords = [chain_coords[i] for i in real_idx]
            chain_c_alpha_coords = [chain_c_alpha_coords[i] for i in real_idx]
            chain_seq = [chain_seq[i] for i in real_idx]
            chain_pure_res_lis = [chain_pure_res_lis[i] for i in real_idx]
            
            embedding_idx = sorted(set(trans.keys()) | set(range(min([i for i in trans.keys() if isinstance(i,int)]),max([i for i in trans.keys() if isinstance(i,int)])+1)),key=standard_residue_sort)
            if lm_embedding_chains is not None:
                try:
                    # print(len(embedding_idx),len(lm_embedding_chain))
                    assert len(embedding_idx) == len(lm_embedding_chain)
                except:
                    raise RuntimeError(peptide_file)
            else:pass
            lm_embedding_chain = [lm_embedding_chain[i] for i in [idx for idx, i in enumerate(embedding_idx) if i in trans.keys()]] if lm_embedding_chains is not None else None
            
            lengths.append(count)
            coords.append(chain_coords)
            
            atoms = []
            trans_ori = {(int(residue.resid) if residue.icode == '' else f"{residue.resid}{residue.icode.strip()}"):res_idx for res_idx, residue in enumerate(peptide_ori.segments[i].residues)}
            real_idx_ori = [trans_ori[idx] for idx in sorted(set(trans_ori.keys()),key=standard_residue_sort)]
            for idx,res in enumerate(peptide_ori.segments[i].residues[real_idx_ori]):
                atoms += [res.atoms.select_atoms(f'name {atom}')[0] for atom in chain.residues[idx].atoms.names]
            new_ori = MDAnalysis.AtomGroup(atoms)
            new_ori.write(os.path.join(
                os.path.dirname(peptide_file), 'peptide_noh.pdb'
            ))
            ori_coords.append(new_ori.positions)
            
            c_alpha_coords.append(np.array(chain_c_alpha_coords))
            seq.append(chain_seq)
            pure_res_lis.append(chain_pure_res_lis)
            valid_lm_embeddings.append(lm_embedding_chain)
            if not count == 0: valid_chain_ids.append(chain.segid)
            
        coords = [atomlist for chainlist in coords for reslist in chainlist for  atomlist in reslist]  # list with n_residues arrays: [n_atoms, 3]
        c_alpha_coords = np.concatenate(c_alpha_coords, axis=0)  # [n_residues, 3]
        lm_embeddings = np.concatenate(valid_lm_embeddings, axis=0) if lm_embedding_chains is not None else None
        seq = np.concatenate(seq, axis=0)
        assert sum(lengths) == len(c_alpha_coords)
        
        new_u_content = ''.join(open(os.path.join(os.path.dirname(peptide_file), 'peptide_noh.pdb'),'r').readlines())
        new_u = MDAnalysis.Universe(StringIO(new_u_content),format='pdb')
        backbone_edge_index,res_atoms_dic = find_backbone_bonds(new_u)
        resid_map = {f'{res.resid}{res.icode}': i for i, res in enumerate(new_u.residues)}
        atom2res_index = [resid_map[f'{atom.resid}{atom.icode}'] for atom in new_u.atoms]
        atom2resid_index = [three2idx[three2self.get(atom.resname, 'X')] for atom in new_u.atoms]
        atom2atomid_index = [atomname2idx[three2idx[three2self.get(atom.resname, 'X')]].get(atom.name,atomname2idx[three2idx[three2self.get(atom.resname, 'X')]]['X']) for atom in new_u.atoms]
        sidechain_edge_index = all_edge_index[~torch.tensor([(i in backbone_edge_index.T.tolist()) for i in all_edge_index.tolist()])]
        all_edge_index = torch.from_numpy(all_edge_index.T)
        sidechain_edge_index = torch.from_numpy(sidechain_edge_index.T)
        if modify is not None:
            pep_a_s = lig_atom_featurizer(modify)
            pep_a_chiralcenters = get_chiralcenters(modify)
        else:
            pep_a_s = lig_atom_featurizer(os.path.join(
                os.path.dirname(peptide_file), 'peptide_noh.pdb'
            ))
            pep_a_chiralcenters = get_chiralcenters(os.path.join(
                os.path.dirname(peptide_file), 'peptide_noh.pdb'
            ))
        pep_a_s = torch.cat([pep_a_s,pep_a_chiralcenters.unsqueeze(-1)],1)
        
        seq = torch.from_numpy(np.asarray(seq)) 
        
        return (new_u_content, torch.from_numpy(np.concatenate(ori_coords)), torch.from_numpy(np.asarray(coords)), lm_embeddings, seq, all_edge_index,backbone_edge_index,sidechain_edge_index, atom2res_index, atom2resid_index, atom2atomid_index, pep_a_s)

def get_updated_peptide_feature(data, device, top_k=24):
    pep_a_pos = data['pep_a'].pos
    atom2atomid_index = data['pep_a'].atom2atomid_index
    
    # Fetch positions based on atomid_index
    n = pep_a_pos[atom2atomid_index==0]
    ca = pep_a_pos[atom2atomid_index==1]
    c = pep_a_pos[atom2atomid_index==2]
    o = pep_a_pos[atom2atomid_index==3]
    tips = pep_a_pos[atom2atomid_index == torch.tensor(aa2tip_idx).to(device)[data['pep_a'].atom2resid_index]]
    
    s,t = torch_cluster.radius_graph(data['pep_a'].atom2res_index.to(torch.float), 1, batch=data['pep_a'].batch,max_num_neighbors=10000)
    index = (data['pep_a'].atom2res_index + data['pep'].ptr[data['pep_a'].batch])[t]
    
    # Vectorize max and min value computation
    diff = (pep_a_pos[s] - pep_a_pos[t]).norm(dim=-1)
    unique_indices, inverse = torch.unique(index, return_inverse=True)
    max_vals = scatter_max(diff, inverse)[0]
    min_vals = scatter_min(diff, inverse)[0]

    # Compute intra_dis
    intra_dis = torch.cat([max_vals.unsqueeze(-1)*0.1,min_vals.unsqueeze(-1)*0.1,(ca-o).norm(dim=-1).unsqueeze(-1)*0.1,(o-n).norm(dim=-1).unsqueeze(-1)*0.1,(n-c).norm(dim=-1).unsqueeze(-1)*0.1],dim=1)
    
    index = data['pep_a'].atom2res_index[atom2atomid_index==0] + data['pep'].ptr[data['pep'].batch]*3
    mask = torch.where(torch.isin(index -1, index), torch.arange(len(index)).to(device), torch.tensor(0).to(device))
    c_before = torch.cat([torch.full((1, 3), float('nan')).to(device),c])[mask]
    mask = torch.where(torch.isin(index +1, index), torch.arange(len(index)).to(device)+2, torch.tensor(0).to(device))
    n_after = torch.cat([torch.full((1, 3), float('nan')).to(device),n])[mask]
    ca_after = torch.cat([torch.full((1, 3), float('nan')).to(device),ca])[mask]

    n_c_before = _normalize(torch.where(torch.logical_and(1.3 < (n - c_before).norm(dim=-1), (n - c_before).norm(dim=-1) < 1.4).unsqueeze(-1).repeat(1,3), (n - c_before), torch.full((1, 3), float('nan')).to(device)))
    ca_n = _normalize(torch.where(torch.logical_and(1.4 < (ca - n).norm(dim=-1), (ca - n).norm(dim=-1) < 1.5).unsqueeze(-1).repeat(1,3), (ca - n), torch.full((1, 3), float('nan')).to(device)))
    c_ca = _normalize(torch.where(torch.logical_and(1.5 < (c-ca).norm(dim=-1), (c-ca).norm(dim=-1) < 1.6).unsqueeze(-1).repeat(1,3), (c-ca), torch.full((1, 3), float('nan')).to(device)))
    n_after_c = _normalize(torch.where(torch.logical_and(1.3 < (n_after-c).norm(dim=-1), (n_after-c).norm(dim=-1) < 1.4).unsqueeze(-1).repeat(1,3), (n_after-c), torch.full((1, 3), float('nan')).to(device)))
    ca_after_n_after = _normalize(torch.where(torch.logical_and(1.4 < (ca_after-n_after).norm(dim=-1), (ca_after-n_after).norm(dim=-1) < 1.5).unsqueeze(-1).repeat(1,3), (ca_after-n_after), torch.full((1, 3), float('nan')).to(device)))

    cosD_phi = torch.sum(_normalize(torch.cross(n_c_before, ca_n), dim=-1) * _normalize(torch.cross(ca_n, c_ca), dim=-1), dim=-1)
    cosD_phi = torch.clamp(cosD_phi, -1 + 1e-7, 1 - 1e-7)
    D_phi = torch.sign(torch.sum(n_c_before * _normalize(torch.cross(ca_n, c_ca), dim=-1), -1)) * torch.acos(cosD_phi) # phi
    cosD_psi = torch.sum(_normalize(torch.cross(ca_n, c_ca), dim=-1) * _normalize(torch.cross(c_ca, n_after_c), dim=-1), dim=-1)
    cosD_psi = torch.clamp(cosD_psi, -1 + 1e-7, 1 - 1e-7)
    D_psi = torch.sign(torch.sum(ca_n * _normalize(torch.cross(c_ca, n_after_c), dim=-1), -1)) * torch.acos(cosD_psi) # psi
    cosD_omega = torch.sum(_normalize(torch.cross(c_ca, n_after_c), dim=-1) * _normalize(torch.cross(n_after_c, ca_after_n_after), dim=-1), dim=-1)
    cosD_omega = torch.clamp(cosD_omega, -1 + 1e-7, 1 - 1e-7)
    D_omega = torch.sign(torch.sum(c_ca * _normalize(torch.cross(n_after_c, ca_after_n_after), dim=-1), -1)) * torch.acos(cosD_omega) # omega

    dihediral_angles = torch.cat([D_phi.unsqueeze(-1),D_psi.unsqueeze(-1),D_omega.unsqueeze(-1)],dim=1) # [n,3]
    
    node_s = torch.cat([intra_dis, dihediral_angles],dim=-1) #[n, 5+3]
    
    center_pos = torch.zeros((len(data['pep'].x), 3)).to(device)
    center_pos.index_add_(0, index=data['pep_a'].atom2res_index + data['pep'].ptr[data['pep_a'].batch], source=pep_a_pos)
    center_pos = center_pos / torch.bincount(data['pep_a'].atom2res_index + data['pep'].ptr[data['pep_a'].batch]).unsqueeze(1)

    center_sidechain_pos = torch.zeros((len(data['pep'].x), 3)).to(device)
    center_sidechain_pos.index_add_(0, index=(data['pep_a'].atom2res_index + data['pep'].ptr[data['pep_a'].batch])[~torch.isin(atom2atomid_index,torch.tensor([0, 2, 3]).to(device))], source=pep_a_pos[~torch.isin(atom2atomid_index,torch.tensor([0, 2, 3]).to(device))])
    center_sidechain_pos = center_sidechain_pos / torch.bincount((data['pep_a'].atom2res_index + data['pep'].ptr[data['pep_a'].batch])[~torch.isin(atom2atomid_index,torch.tensor([0, 2, 3]).to(device))]).unsqueeze(1)
    
    edge_index = torch_cluster.knn_graph(x=ca, k=top_k, batch=data['pep'].batch)
    
    minmax = torch.zeros_like(edge_index).T

    cadist = (torch.pairwise_distance(ca[edge_index[0]], ca[edge_index[1]]) * 0.1).view(-1,1)
    cedist = (torch.pairwise_distance(center_pos[edge_index[0]], center_pos[edge_index[1]]) * 0.1).view(-1,1)
    csdist = (torch.pairwise_distance(center_sidechain_pos[edge_index[0]], center_sidechain_pos[edge_index[1]]) * 0.1).view(-1,1)
    
    edge_connect = (torch.abs(edge_index[0] - edge_index[1]) == 1 & torch.all(n_c_before[torch.max(edge_index,dim=0)[0]] != 0)).long()

    positional_embedding = positional_embeddings_v1(edge_index,device=device)
    E_vectors = ca[edge_index[0]] - ca[edge_index[1]] # [n_edges, 3]
    edge_s = torch.cat([edge_connect.unsqueeze(-1), cadist, cedist, csdist, minmax, _rbf(E_vectors.norm(dim=-1), D_count=16, device=device), positional_embedding], dim=1) # 1+1+1+1+16+81
    edge_v = _normalize(E_vectors).unsqueeze(-2)
    
    return node_s, ca, tips, edge_index, edge_s, edge_v
