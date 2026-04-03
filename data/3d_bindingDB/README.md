This section includes experimental results obtained using the 3D BindingDB dataset constructed in TUNA.

## How to load amino acid sequence and SMILES

If you want to retrieve the protein sequence and ligand SMILES from the provided feature files, you can load the dictionaries as follows:

```python
import numpy as np

seq_dict = np.load('./F_AASeq.npy', allow_pickle=True)
sequence = seq_dict.item()[pid]

smi_dict = np.load('./F_AASMI.npy', allow_pickle=True)
smiles = smi_dict.item()[cid]
