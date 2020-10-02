#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('conda install -y -c rdkit rdkit')


# In[ ]:


pip install --pre dgl


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
import numpy as np
import pandas as pd
from itertools import combinations
from glob import glob

from rdkit.Chem import rdmolops, ChemicalFeatures

#from utils import print_progress
#import constants as C

import os
get_ipython().run_line_magic('cd', '/kaggle/input/champs-scalar-coupling')
print(os.listdir("."))

# Any results you write to the current directory are saved as output.


# In[ ]:


##
# Copied from: https://github.com/jensengroup/xyz2mol
# 
#
# Written by Jan H. Jensen based on this paper Yeonjoon Kim and Woo Youn Kim
# "Universal Structure Conversion Method for Organic Molecules: From Atomic Connectivity
# to Three-Dimensional Geometry" Bull. Korean Chem. Soc. 2015, Vol. 36, 1769-1777 DOI: 10.1002/bkcs.10334
#
from rdkit import Chem
from rdkit.Chem import AllChem
import itertools
from rdkit.Chem import rdmolops
from collections import defaultdict
import copy
import networkx as nx #uncomment if you don't want to use "quick"/install networkx


global __ATOM_LIST__
__ATOM_LIST__ = [ x.strip() for x in ['h ','he',       'li','be','b ','c ','n ','o ','f ','ne',       'na','mg','al','si','p ','s ','cl','ar',       'k ','ca','sc','ti','v ','cr','mn','fe','co','ni','cu',       'zn','ga','ge','as','se','br','kr',       'rb','sr','y ','zr','nb','mo','tc','ru','rh','pd','ag',       'cd','in','sn','sb','te','i ','xe',       'cs','ba','la','ce','pr','nd','pm','sm','eu','gd','tb','dy',       'ho','er','tm','yb','lu','hf','ta','w ','re','os','ir','pt',       'au','hg','tl','pb','bi','po','at','rn',       'fr','ra','ac','th','pa','u ','np','pu'] ]


def get_atom(atom):
    global __ATOM_LIST__
    atom = atom.lower()
    return __ATOM_LIST__.index(atom) + 1


def getUA(maxValence_list, valence_list):
    UA = []
    DU = []
    for i, (maxValence,valence) in enumerate(zip(maxValence_list, valence_list)):
        if maxValence - valence > 0:
            UA.append(i)
            DU.append(maxValence - valence)
    return UA,DU


def get_BO(AC,UA,DU,valences,UA_pairs,quick):
    BO = AC.copy()
    DU_save = []

    while DU_save != DU:
        for i,j in UA_pairs:
            BO[i,j] += 1
            BO[j,i] += 1

        BO_valence = list(BO.sum(axis=1))
        DU_save = copy.copy(DU)
        UA, DU = getUA(valences, BO_valence)
        UA_pairs = get_UA_pairs(UA,AC,quick)[0]

    return BO


def valences_not_too_large(BO,valences):
    number_of_bonds_list = BO.sum(axis=1)
    for valence, number_of_bonds in zip(valences,number_of_bonds_list):
        if number_of_bonds > valence:
            return False

    return True


def BO_is_OK(BO,AC,charge,DU,atomic_valence_electrons,atomicNumList,charged_fragments):
    Q = 0 # total charge
    q_list = []
    if charged_fragments:
        BO_valences = list(BO.sum(axis=1))
        for i,atom in enumerate(atomicNumList):
            q = get_atomic_charge(atom,atomic_valence_electrons[atom],BO_valences[i])
            Q += q
            if atom == 6:
                number_of_single_bonds_to_C = list(BO[i,:]).count(1)
                if number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:
                    Q += 1
                    q = 2
                if number_of_single_bonds_to_C == 3 and Q + 1 < charge:
                    Q += 2
                    q = 1

            if q != 0:
                q_list.append(q)

    if (BO-AC).sum() == sum(DU) and charge == Q and len(q_list) <= abs(charge):
        return True
    else:
        return False


def get_atomic_charge(atom,atomic_valence_electrons,BO_valence):
    if atom == 1:
        charge = 1 - BO_valence
    elif atom == 5:
        charge = 3 - BO_valence
    elif atom == 15 and BO_valence == 5:
        charge = 0
    elif atom == 16 and BO_valence == 6:
        charge = 0
    else:
        charge = atomic_valence_electrons - 8 + BO_valence

    return charge

def clean_charges(mol):
    # this hack should not be needed any more but is kept just in case

    rxn_smarts = ['[N+:1]=[*:2]-[C-:3]>>[N+0:1]-[*:2]=[C-0:3]',
                  '[N+:1]=[*:2]-[O-:3]>>[N+0:1]-[*:2]=[O-0:3]',
                  '[N+:1]=[*:2]-[*:3]=[*:4]-[O-:5]>>[N+0:1]-[*:2]=[*:3]-[*:4]=[O-0:5]',
                  '[#8:1]=[#6:2]([!-:6])[*:3]=[*:4][#6-:5]>>[*-:1][*:2]([*:6])=[*:3][*:4]=[*+0:5]',
                  '[O:1]=[c:2][c-:3]>>[*-:1][*:2][*+0:3]',
                  '[O:1]=[C:2][C-:3]>>[*-:1][*:2]=[*+0:3]']

    fragments = Chem.GetMolFrags(mol,asMols=True,sanitizeFrags=False)

    for i,fragment in enumerate(fragments):
        for smarts in rxn_smarts:
            patt = Chem.MolFromSmarts(smarts.split(">>")[0])
            while fragment.HasSubstructMatch(patt):
                rxn = AllChem.ReactionFromSmarts(smarts)
                ps = rxn.RunReactants((fragment,))
                fragment = ps[0][0]
        if i == 0:
            mol = fragment
        else:
            mol = Chem.CombineMols(mol,fragment)

    return mol


def BO2mol(mol,BO_matrix, atomicNumList,atomic_valence_electrons,mol_charge,charged_fragments):
    # based on code written by Paolo Toscani

    l = len(BO_matrix)
    l2 = len(atomicNumList)
    BO_valences = list(BO_matrix.sum(axis=1))

    if (l != l2):
        raise RuntimeError('sizes of adjMat ({0:d}) and atomicNumList '
            '{1:d} differ'.format(l, l2))

    rwMol = Chem.RWMol(mol)

    bondTypeDict = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE
    }

    for i in range(l):
        for j in range(i + 1, l):
            bo = int(round(BO_matrix[i, j]))
            if (bo == 0):
                continue
            bt = bondTypeDict.get(bo, Chem.BondType.SINGLE)
            rwMol.AddBond(i, j, bt)
    mol = rwMol.GetMol()

    if charged_fragments:
        mol = set_atomic_charges(mol,atomicNumList,atomic_valence_electrons,BO_valences,BO_matrix,mol_charge)
    else:
        mol = set_atomic_radicals(mol,atomicNumList,atomic_valence_electrons,BO_valences)

    return mol

def set_atomic_charges(mol,atomicNumList,atomic_valence_electrons,BO_valences,BO_matrix,mol_charge):
    q = 0
    for i,atom in enumerate(atomicNumList):
        a = mol.GetAtomWithIdx(i)
        charge = get_atomic_charge(atom,atomic_valence_electrons[atom],BO_valences[i])
        q += charge
        if atom == 6:
            number_of_single_bonds_to_C = list(BO_matrix[i,:]).count(1)
            if number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:
                    q += 1
                    charge = 0
            if number_of_single_bonds_to_C == 3 and q + 1 < mol_charge:
                    q += 2
                    charge = 1

        if (abs(charge) > 0):
            a.SetFormalCharge(int(charge))

    # shouldn't be needed anymore bit is kept just in case
    #mol = clean_charges(mol)

    return mol


def set_atomic_radicals(mol,atomicNumList,atomic_valence_electrons,BO_valences):
    # The number of radical electrons = absolute atomic charge
    for i,atom in enumerate(atomicNumList):
        a = mol.GetAtomWithIdx(i)
        charge = get_atomic_charge(atom,atomic_valence_electrons[atom],BO_valences[i])

        if (abs(charge) > 0):
            a.SetNumRadicalElectrons(abs(int(charge)))

    return mol

def get_bonds(UA,AC):
    bonds = []

    for k,i in enumerate(UA):
        for j in UA[k+1:]:
            if AC[i,j] == 1:
                bonds.append(tuple(sorted([i,j])))

    return bonds

def get_UA_pairs(UA,AC,quick):
    bonds = get_bonds(UA,AC)
    if len(bonds) == 0:
        return [()]

    if quick:
        G=nx.Graph()
        G.add_edges_from(bonds)
        UA_pairs = [list(nx.max_weight_matching(G))]
        return UA_pairs

    max_atoms_in_combo = 0
    UA_pairs = [()]
    for combo in list(itertools.combinations(bonds, int(len(UA)/2))):
        flat_list = [item for sublist in combo for item in sublist]
        atoms_in_combo = len(set(flat_list))
        if atoms_in_combo > max_atoms_in_combo:
            max_atoms_in_combo = atoms_in_combo
            UA_pairs = [combo]
 #           if quick and max_atoms_in_combo == 2*int(len(UA)/2):
 #               return UA_pairs
        elif atoms_in_combo == max_atoms_in_combo:
            UA_pairs.append(combo)

    return UA_pairs

def AC2BO(AC,atomicNumList,charge,charged_fragments,quick):
    # TODO
    atomic_valence = defaultdict(list)
    atomic_valence[1] = [1]
    atomic_valence[6] = [4]
    atomic_valence[7] = [4,3]
    atomic_valence[8] = [2,1]
    atomic_valence[9] = [1]
    atomic_valence[14] = [4]
    atomic_valence[15] = [5,4,3]
    atomic_valence[16] = [6,4,2]
    atomic_valence[17] = [1]
    atomic_valence[32] = [4]
    atomic_valence[35] = [1]
    atomic_valence[53] = [1]


    atomic_valence_electrons = {}
    atomic_valence_electrons[1] = 1
    atomic_valence_electrons[6] = 4
    atomic_valence_electrons[7] = 5
    atomic_valence_electrons[8] = 6
    atomic_valence_electrons[9] = 7
    atomic_valence_electrons[14] = 4
    atomic_valence_electrons[15] = 5
    atomic_valence_electrons[16] = 6
    atomic_valence_electrons[17] = 7
    atomic_valence_electrons[32] = 4
    atomic_valence_electrons[35] = 7
    atomic_valence_electrons[53] = 7

    # make a list of valences, e.g. for CO: [[4],[2,1]]
    valences_list_of_lists = []
    for atomicNum in atomicNumList:
        valences_list_of_lists.append(atomic_valence[atomicNum])

    # convert [[4],[2,1]] to [[4,2],[4,1]]
    valences_list = list(itertools.product(*valences_list_of_lists))

    best_BO = AC.copy()

    # implemenation of algorithm shown in Figure 2
    # UA: unsaturated atoms
    # DU: degree of unsaturation (u matrix in Figure)
    # best_BO: Bcurr in Figure
    #

    for valences in valences_list:
        AC_valence = list(AC.sum(axis=1))
        UA,DU_from_AC = getUA(valences, AC_valence)

        if len(UA) == 0 and BO_is_OK(AC,AC,charge,DU_from_AC,atomic_valence_electrons,atomicNumList,charged_fragments):
            return AC,atomic_valence_electrons

        UA_pairs_list = get_UA_pairs(UA,AC,quick)
        for UA_pairs in UA_pairs_list:
            BO = get_BO(AC,UA,DU_from_AC,valences,UA_pairs,quick)
            if BO_is_OK(BO,AC,charge,DU_from_AC,atomic_valence_electrons,atomicNumList,charged_fragments):
                return BO,atomic_valence_electrons

            elif BO.sum() >= best_BO.sum() and valences_not_too_large(BO,valences):
                best_BO = BO.copy()

    return best_BO,atomic_valence_electrons


def AC2mol(mol,AC,atomicNumList,charge,charged_fragments,quick):
    # convert AC matrix to bond order (BO) matrix
    BO,atomic_valence_electrons = AC2BO(AC,atomicNumList,charge,charged_fragments,quick)

    # add BO connectivity and charge info to mol object
    mol = BO2mol(mol,BO, atomicNumList,atomic_valence_electrons,charge,charged_fragments)

    return mol


def get_proto_mol(atomicNumList):
    mol = Chem.MolFromSmarts("[#"+str(atomicNumList[0])+"]")
    rwMol = Chem.RWMol(mol)
    for i in range(1,len(atomicNumList)):
        a = Chem.Atom(atomicNumList[i])
        rwMol.AddAtom(a)

    mol = rwMol.GetMol()

    return mol


def get_atomicNumList(atomic_symbols):
    atomicNumList = []
    for symbol in atomic_symbols:
        atomicNumList.append(get_atom(symbol))
    return atomicNumList


def read_xyz_file(filename):

    atomic_symbols = []
    xyz_coordinates = []

    with open(filename, "r") as file:
        for line_number,line in enumerate(file):
            if line_number == 0:
                num_atoms = int(line)
            elif line_number == 1:
                if "charge=" in line:
                    charge = int(line.split("=")[1])
                else:
                    charge = 0
            else:
                atomic_symbol, x, y, z = line.split()
                atomic_symbols.append(atomic_symbol)
                xyz_coordinates.append([float(x),float(y),float(z)])

    atomicNumList = get_atomicNumList(atomic_symbols)

    return atomicNumList,charge,xyz_coordinates

def xyz2AC(atomicNumList,xyz):
    import numpy as np
    mol = get_proto_mol(atomicNumList)

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i,(xyz[i][0],xyz[i][1],xyz[i][2]))
    mol.AddConformer(conf)

    dMat = Chem.Get3DDistanceMatrix(mol)
    pt = Chem.GetPeriodicTable()

    num_atoms = len(atomicNumList)
    AC = np.zeros((num_atoms,num_atoms)).astype(int)

    for i in range(num_atoms):
        a_i = mol.GetAtomWithIdx(i)
        Rcov_i = pt.GetRcovalent(a_i.GetAtomicNum())*1.30
        for j in range(i+1,num_atoms):
            a_j = mol.GetAtomWithIdx(j)
            Rcov_j = pt.GetRcovalent(a_j.GetAtomicNum())*1.30
            if dMat[i,j] <= Rcov_i + Rcov_j:
                AC[i,j] = 1
                AC[j,i] = 1

    return AC,mol,dMat

def chiral_stereo_check(mol):
    Chem.DetectBondStereochemistry(mol,-1)
    Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
    Chem.AssignAtomChiralTagsFromStructure(mol,-1)
    return mol

def xyz2mol(atomicNumList, charge, xyz_coordinates, charged_fragments, quick,
            check_chiral_stereo=True):

    # Get atom connectivity (AC) matrix, list of atomic numbers, molecular charge,
    # and mol object with no connectivity information
    AC,mol,dMat = xyz2AC(atomicNumList, xyz_coordinates)

    # Convert AC to bond order matrix and add connectivity and charge info to mol object
    new_mol = AC2mol(mol, AC, atomicNumList, charge, charged_fragments, quick)

    # sanitize
    try: Chem.SanitizeMol(new_mol)
    except ValueError as e: print(e)

    # Check for stereocenters and chiral centers
    if check_chiral_stereo:
        try: new_mol = chiral_stereo_check(new_mol)
        except ValueError as e: print(e)

    return new_mol,dMat


# In[ ]:


import os
get_ipython().run_line_magic('cd', '/kaggle/input/champs-scalar-coupling')
print(os.listdir("."))


# In[ ]:


import gc
import numpy as np
import pandas as pd
from itertools import combinations
from glob import glob

from rdkit.Chem import rdmolops, ChemicalFeatures



mol_feat_columns = ['ave_bond_length', 'std_bond_length', 'ave_atom_weight']
xyz_filepath_list = list(glob('structures/*.xyz'))
xyz_filepath_list.sort()


## Functions to create the RDKit mol objects
def mol_from_xyz(filepath, add_hs=True, compute_dist_centre=False):
    """Wrapper function for calling xyz2mol function."""
    charged_fragments = True  # alternatively radicals are made

    # quick is faster for large systems but requires networkx
    # if you don't want to install networkx set quick=False and
    # uncomment 'import networkx as nx' at the top of the file
    quick = True

    atomicNumList, charge, xyz_coordinates = read_xyz_file(filepath)
    mol, dMat = xyz2mol(atomicNumList, charge, xyz_coordinates,
                        charged_fragments, quick, check_chiral_stereo=False)

    return mol, np.array(xyz_coordinates), dMat


# In[ ]:


mol, xyz_coordinates, dMat = mol_from_xyz(xyz_filepath_list[0])


# mol: rdkit.Chem.rdchem.Mol
# dMat: Distance Matrix

# In[ ]:


from dgl.data.chem import mol_to_bigraph


# In[ ]:


g = mol_to_bigraph(mol)


# In[ ]:


from dgl.data.chem import mol_to_bigraph
def get_graph(filepath='./', add_hs=True, compute_dist_centre=False):
    graph_list = []
    xyz_filepath_list = list(glob(filepath+'structures/*.xyz'))
    xyz_filepath_list.sort()
    for i in range(len(xyz_filepath_list)):
        mol, xyz_coordinates, _ = mol_from_xyz(xyz_filepath_list[i])
        try:
            g = mol_to_bigraph(mol)
            g.ndata['xyz'] = xyz_coordinates
            graph_list.append(g)
        except RuntimeError:
            pass
            
    return graph_list


# In[ ]:


def get_molecules():
    """
    Constructs rdkit mol objects derrived from the .xyz files. Also returns:
        - mol ids (unique numerical ids)
        - set of molecule level features
        - arrays of xyz coordinates
        - euclidean distance matrices
        - graph distance matrices.
    All objects are returned in dictionaries with 'mol_name' as keys.
    """
    N_MOLS =  130775
    MAX_N_ATOMS = 29
    mols, mol_ids, mol_feats = {}, {}, {}
    xyzs, dist_matrices, graph_dist_matrices = {}, {}, {}
    print('Create molecules and distance matrices.')
    for i in range(N_MOLS):
        #print_progress(i, N_MOLS)
        filepath = xyz_filepath_list[i]
        mol_name = filepath.split('/')[-1][:-4]
        mol, xyz, dist_matrix = mol_from_xyz(filepath)
        mols[mol_name] = mol
        xyzs[mol_name] = xyz
        dist_matrices[mol_name] = dist_matrix
        mol_ids[mol_name] = i

        # make padded graph distance matrix dataframes
        n_atoms = len(xyz)
        graph_dist_matrix = pd.DataFrame(np.pad(
            rdmolops.GetDistanceMatrix(mol),
            [(0, 0), (0, MAX_N_ATOMS - n_atoms)], 'constant'
        ))
        graph_dist_matrix['molecule_id'] = n_atoms * [i]
        graph_dist_matrices[mol_name] = graph_dist_matrix

        # compute molecule level features
        adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
        atomic_num_list, _, _ = read_xyz_file(filepath)
        dists = dist_matrix.ravel()[np.tril(adj_matrix).ravel()==1]
        mol_feats[mol_name] = pd.Series(
            [np.mean(dists), np.std(dists), np.mean(atomic_num_list)],
            index=mol_feat_columns
        )

    return mols, mol_ids, mol_feats, xyzs, dist_matrices, graph_dist_matrices


# In[ ]:


from dgl.data.chem import mol_to_bigraph
import torch as th

def get_molecules(filepath='./', add_hs=True, compute_dist_centre=False):
    N_MOLS =  130775
    MAX_N_ATOMS = 29
    graph_list = []
    mol_feat_columns = ['ave_bond_length', 'std_bond_length', 'ave_atom_weight']

    xyz_filepath_list = list(glob(filepath+'structures/*.xyz'))
    xyz_filepath_list.sort()
    
    for i in range(100):
        mol, xyz, dist_matrix = mol_from_xyz(xyz_filepath_list[i])
        try:
            g = mol_to_bigraph(mol)
            
            g.ndata['xyz'] = xyz
            g.gdata = {}
            
            mol_name = xyz_filepath_list[i].split('/')[-1][:-4]
            g.gdata['name'] = mol_name 
            
            g.gdata['dist_matrices'] = th.tensor(dist_matrix)
            g.gdata['mol_id'] = i
            
            g.ndata['atom_ids'] = th.tensor((len(xyz) * [i]))
            graph_list.append(g)
            
            graph_dist_matrix = pd.DataFrame(np.pad(
            rdmolops.GetDistanceMatrix(mol),
            [(0, 0), (0, MAX_N_ATOMS - len(xyz))], 'constant'))
            g.gdata['graph_dist_matrix'] = graph_dist_matrix
            
            # compute molecule level features
            adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
            atomic_num_list, _, _ = read_xyz_file(xyz_filepath_list[i])
            dists = dist_matrix.ravel()[np.tril(adj_matrix).ravel()==1]
            g.gdata['mol_feat'] = pd.Series(
            [np.mean(dists), np.std(dists), np.mean(atomic_num_list)],
            index=mol_feat_columns
        )
        except RuntimeError:
            pass
            
    return graph_list


# In[ ]:


graph_list = get_molecules()


# In[ ]:


import numpy as np
import pandas as pd
import pickle

from dgl.data.chem.utils import mol_to_bigraph,                                CanonicalAtomFeaturizer
import torch
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures, rdmolops
from rdkit import RDConfig

from glob import glob

import os.path as osp
from glob import glob
from functools import partial
from collections import defaultdict
from dgl import DGLGraph

xyz_filepath_list = list(glob('structures/*.xyz'))
xyz_filepath_list.sort()


## Functions to create the RDKit mol objects
def mol_from_xyz(filepath, add_hs=True, compute_dist_centre=False):
    """Wrapper function for calling xyz2mol function."""
    charged_fragments = True  # alternatively radicals are made

    # quick is faster for large systems but requires networkx
    # if you don't want to install networkx set quick=False and
    # uncomment 'import networkx as nx' at the top of the file
    quick = True

    atomicNumList, charge, xyz_coordinates = read_xyz_file(filepath)
    mol, dMat = xyz2mol(atomicNumList, charge, xyz_coordinates,
                        charged_fragments, quick, check_chiral_stereo=False)

    return mol, np.array(xyz_coordinates), dMat

def my_mol_to_graph(mol, graph_constructor, atom_featurizer, bond_featurizer):
    """Convert an RDKit molecule object into a DGLGraph and featurize for it.
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    graph_constructor : callable
        Takes an RDKit molecule as input and returns a DGLGraph
    atom_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for atoms in a molecule, which can be used to update
        ndata for a DGLGraph.
    bond_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for bonds in a molecule, which can be used to update
        edata for a DGLGraph.
    Returns
    -------
    g : DGLGraph
        Converted DGLGraph for the molecule
    """
    #new_order = rdmolfiles.CanonicalRankAtoms(mol)
    #mol = rdmolops.RenumberAtoms(mol, new_order)
    g = graph_constructor(mol)

    if atom_featurizer is not None:
        g.ndata.update(atom_featurizer(mol))

    if bond_featurizer is not None:
        g.edata.update(bond_featurizer(mol))

    return g


def construct_bigraph_from_mol(mol, add_self_loop=False):
    """Construct a bi-directed DGLGraph with topology only for the molecule.
    The **i** th atom in the molecule, i.e. ``mol.GetAtomWithIdx(i)``, corresponds to the
    **i** th node in the returned DGLGraph.
    The **i** th bond in the molecule, i.e. ``mol.GetBondWithIdx(i)``, corresponds to the
    **(2i)**-th and **(2i+1)**-th edges in the returned DGLGraph. The **(2i)**-th and
    **(2i+1)**-th edges will be separately from **u** to **v** and **v** to **u**, where
    **u** is ``bond.GetBeginAtomIdx()`` and **v** is ``bond.GetEndAtomIdx()``.
    If self loops are added, the last **n** edges will separately be self loops for
    atoms ``0, 1, ..., n-1``.
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs.
    Returns
    -------
    g : DGLGraph
        Empty bigraph topology of the molecule
    """
    g = DGLGraph()

    # Add nodes
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    # Add edges
    src_list = []
    dst_list = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])
    g.add_edges(src_list, dst_list)

    if add_self_loop:
        nodes = g.nodes()
        g.add_edges(nodes, nodes)

    return g


def my_mol_to_bigraph(mol, add_self_loop=False,
                   atom_featurizer=CanonicalAtomFeaturizer(),
                   bond_featurizer=None):
    """Convert an RDKit molecule object into a bi-directed DGLGraph and featurize for it.
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs.
    atom_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to CanonicalAtomFeaturizer().
    bond_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for bonds in a molecule, which can be used to update
        edata for a DGLGraph.
    Returns
    -------
    g : DGLGraph
        Bi-directed DGLGraph for the molecule
    """
    return my_mol_to_graph(mol, partial(construct_bigraph_from_mol, add_self_loop=add_self_loop),
                        atom_featurizer, bond_featurizer)
  
def bond_featurizer(mol, self_loop=True):
    """Featurization for all bonds in a molecule.
    The bond indices will be preserved.
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object
    self_loop : bool
        Whether to add self loops. Default to be False.
    Returns
    -------
    bond_feats_dict : dict
        Dictionary for bond features
    """
    bond_feats_dict = defaultdict(list)

    mol_conformers = mol.GetConformers()
    assert len(mol_conformers) == 1
    geom = mol_conformers[0].GetPositions()

    num_atoms = mol.GetNumAtoms()
    for u in range(num_atoms):
        for v in range(num_atoms):
            if u == v and not self_loop:
                continue

            e_uv = mol.GetBondBetweenAtoms(u, v)
            if e_uv is None:
                continue
            else:
                bond_type = e_uv.GetBondType()
            bond_feats_dict['e_feat'].append([
                float(bond_type == x)
                for x in (Chem.rdchem.BondType.SINGLE,
                          Chem.rdchem.BondType.DOUBLE,
                          Chem.rdchem.BondType.TRIPLE,
                          Chem.rdchem.BondType.AROMATIC)
            ])
            bond_feats_dict['distance'].append(
                np.linalg.norm(geom[u] - geom[v]))

    bond_feats_dict['e_feat'] = torch.tensor(
        np.array(bond_feats_dict['e_feat']).astype(np.float32))
    bond_feats_dict['distance'] = torch.tensor(
        np.array(bond_feats_dict['distance']).astype(np.float32)).reshape(-1 , 1)

    return bond_feats_dict

class KaggleMolDataset(object):
    def __init__(self, 
                 file_list = xyz_filepath_list,
                 label_filepath = './',
                 store_path = './' ,
                 mode='train', 
                 from_raw=True,
                 mol_to_graph = my_mol_to_bigraph,
                 atom_featurizer=CanonicalAtomFeaturizer,
                 bond_featurizer=bond_featurizer):

        assert mode in ['train', 'test'],            'Expect mode to be train or test, got {}.'.format(mode)

        self.mode = mode
        
        self.from_raw = from_raw
        """
        if not from_raw:
            file_name = "%s_processed" % (mode)
        else:
            file_name = "structures"
        self.file_dir = pathlib.Path(file_dir, file_name)
        """
        self.file_list = file_list
        self.store_path = store_path
        self.label_filepath = label_filepath
        self.graphs, self.labels = [],[]
        self._load(mol_to_graph, atom_featurizer, bond_featurizer)

    def _load(self, mol_to_graph, atom_featurizer, bond_featurizer):
        if not self.from_raw:
            pass
        #    with open(osp.join(self.file_dir, "%s_graphs.pkl" % self.mode), "rb") as f:
        #        self.graphs = pickle.load(f)
        #    with open(osp.join(self.file_dir, "%s_labels.pkl" % self.mode), "rb") as f:
        #        self.labels = pickle.load(f)
        else:
            print('Start preprocessing dataset...')
            labels  = pd.read_csv(self.label_filepath +self.mode + '.csv')
            cnt = 0
            dataset_size = len(labels['molecule_name'].unique())
            mol_names = labels['molecule_name'].unique()
            
            for i in range(len(self.file_list)):
                mol_name = self.file_list[i].split('/')[-1][:-4] 
                if mol_name in mol_names:
                    cnt += 1
                    if cnt%100 ==0:
                        print('Processing molecule {:d}/{:d}'.format(cnt, dataset_size))
                    mol, xyz, dist_matrix = mol_from_xyz(self.file_list[i])
                    Chem.SanitizeMol(mol)
                    graph = mol_to_graph(mol, bond_featurizer=bond_featurizer)  
                    graph.gdata = {}    
                    smiles = Chem.MolToSmiles(mol)
                    graph.gdata['smiles'] = smiles    
                    graph.gdata['mol_name'] = mol_name 
                    graph.ndata['h'] = torch.cat([graph.ndata['h'], torch.tensor(xyz).float()],
                                                  dim = 1)
                    self.graphs.append(graph)
                    label = labels[labels['molecule_name'] ==mol_name ].drop([
                                                                        'molecule_name', 
                                                                        'type',
                                                                        'id'
                                                                      ],
                                                                         axis = 1
                                                                    )
                    self.labels.append(label)

            with open("%s_grapgs.pkl" % self.mode, "wb") as f:
                pickle.dump(self.graphs, f)
            with open( "%s_labels.pkl" % self.mode, "wb") as f:
                pickle.dump(self.labels, f)

        print(len(self.graphs), "loaded!")

    def __getitem__(self, item):
        """Get datapoint with index
        Parameters
        ----------
        item : int
            Datapoint index
        Returns
        -------
        str
            SMILES for the ith datapoint
        DGLGraph
            DGLGraph for the ith datapoint
        Tensor of dtype float32
            Labels of the datapoint for all tasks
        """
        g, l = self.graphs[item], self.labels[item]
        return g.smile, g, l

    def __len__(self):
        """Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        """
        return len(self.graphs)


# In[ ]:


dataset = KaggleMolDataset(mode = 'test')


# In[ ]:


train =  KaggleMolDataset()

