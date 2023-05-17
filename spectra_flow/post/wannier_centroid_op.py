from typing import Dict, List, Tuple, Union
from pathlib import Path
import dpdata, numpy as np
from dflow.python import (
    OP, 
    OPIO, 
    Artifact, 
    OPIOSign, 
    BigParameter,
)
from dflow.utils import (
    set_directory
)
from spectra_flow.post.cal_wannier_centroid import cal_wc_h2o
from spectra_flow.utils import read_conf

class CalWC(OP):
    def __init__(self) -> None:
        super().__init__()
    
    @classmethod
    def get_input_sign(cls):
        return OPIOSign({
            "confs": Artifact(Path),
            "conf_fmt": BigParameter(dict),
            "cal_wc_python": Artifact(Path, optional = True),
            "wannier_function_centers": Artifact(Dict[str, Path]),
        })

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({
            "wannier_centroid": Artifact(Dict[str, Path])
        })

    @OP.exec_sign_check
    def execute(
            self,
            op_in: OPIO,
    ) -> OPIO:
        confs = read_conf(op_in["confs"], op_in["conf_fmt"])
        if op_in["cal_wc_python"]:
            import imp
            cal_wc_python = imp.load_source("wc_module", str(op_in["cal_wc_python"]))
            cal_wc = cal_wc_python.cal_wc
        else:
            cal_wc = self.cal_wc
        wfc_d = op_in["wannier_function_centers"]
        wc_d = {}
        for key in wfc_d:
            wfc = np.loadtxt(wfc_d[key], dtype = float, ndmin = 2)
            wc = cal_wc(confs, wfc)
            wc_path = Path(f"wc_{key}.raw")
            np.savetxt(wc_path, wc, fmt = "%15.8f")
            wc_d[key] = wc_path
        return OPIO({
            "wannier_centroid": wc_d
        })
    
    def cal_wc(self, confs: dpdata.System, wfc: np.ndarray) -> np.ndarray:
        return cal_wc_h2o(
            wfc.reshape(confs.get_nframes(), -1, 3), 
            confs["coords"][:, confs["atom_types"] == 0], 
            np.diagonal(confs["cells"], axis1 = -2, axis2 = -1)
        ).reshape(confs.get_nframes(), -1)
