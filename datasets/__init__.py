# P2R_ZIP/datasets/__init__.py
from .jhu import JHU_Crowd
from .shha import SHHA
from .ucf_qnrf import UCF_QNRF
from .nwpu import NWPU

def get_dataset(name):
    name = name.lower()
    
    if name == "jhu": return JHU_Crowd
    
    if name == "shha": return SHHA
    if name == "shhb": return SHHA  
    
    if name == "ucf": return UCF_QNRF
    if name == "nwpu": return NWPU
    
    raise ValueError(f"Dataset {name} non riconosciuto.")