from ._base import Vanilla
from .KD import KD
from .FitNet import FitNet
from .CRD import CRD
from .DKD import DKD
from .WKD import WKD
from .New import NEW
from .FFD import FFD
from .LBL import LBL

distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "FITNET": FitNet,
    "CRD": CRD,
    "DKD": DKD,
    "WKD": WKD,
    'NEW': NEW,
    'FFD': FFD,
    'LBL': LBL,
}
