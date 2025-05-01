from typing import Optional
import numpy as np
from spectra_flow.utils import (
    calculate_corr,
    get_distance,
    apply_lorenz_filter,
    apply_gaussian_filter,
    FT,
    FT_fft
)

# def calculate_corr_polar(polar: np.ndarray, window: int):
#     polar_iso = np.mean(polar.diagonal(offset = 0, axis1 = 1, axis2 = 2), axis = 1)

#     diag = np.zeros((polar_iso.shape[0], 3, 3), dtype = float)
#     diag[:, 0, 0] = polar_iso
#     diag[:, 1, 1] = polar_iso
#     diag[:, 2, 2] = polar_iso
#     polar_aniso = polar - diag # type: ignore

#     polar_iso -= np.mean(polar_iso, axis = 0, keepdims = True)
#     polar_aniso -= np.mean(polar_aniso, axis = 0, keepdims = True)

#     corr_iso = np.sum(calculate_corr(polar_iso, polar_iso, window), axis = -1)
#     polar_aniso = polar_aniso.reshape(-1, 9)
#     corr_aniso = np.sum(calculate_corr(polar_aniso, polar_aniso, window), axis = -1)
#     corr_aniso *= 2 / 15
#     return corr_iso, corr_aniso

def calculate_corr_polar(atomic_polar: np.ndarray, coords: np.ndarray, cells: np.ndarray, window: int):
    nframes, natom = atomic_polar.shape[:2]

    polar_iso = np.mean(atomic_polar.diagonal(offset = 0, axis1 = -2, axis2 = -1), axis = -1)
    diag = np.zeros_like(atomic_polar, dtype = float)
    diag[..., 0, 0] = polar_iso
    diag[..., 1, 1] = polar_iso
    diag[..., 2, 2] = polar_iso
    polar_aniso = atomic_polar - diag # type: ignore

    polar_iso -= np.mean(polar_iso, axis = 0, keepdims = True)
    polar_aniso -= np.mean(polar_aniso, axis = 0, keepdims = True)
    polar_aniso = polar_aniso.reshape(nframes, natom, 9)

    corr_iso_intra = calculate_corr(polar_iso, polar_iso, window)
    corr_aniso_intra = np.sum(calculate_corr(polar_aniso, polar_aniso, window), axis = -1) * (2. / 15.)                   

    polar_iso_cutoff = np.empty_like(polar_iso)
    polar_aniso_cutoff = np.empty_like(polar_aniso)

    for atom_i in range(natom):
      #dis_mask = np.ones((nframes, natom, 1), dtype=bool)  # 直接创建正确的形状
      dis_mask = get_distance(coords, coords[:, [atom_i], :], cells) >=0
      dis_mask[:, atom_i] = False
      polar_iso_cutoff[:, atom_i] = np.matmul(polar_iso[:, None, :], dis_mask).squeeze((1, 2))
      polar_aniso_cutoff[:, atom_i] = np.matmul(polar_aniso.transpose(0, 2, 1), dis_mask).squeeze(2)

    corr_iso_inter = calculate_corr(polar_iso_cutoff, polar_iso, window)
    corr_aniso_inter = np.sum(calculate_corr(polar_aniso_cutoff, polar_aniso, window), axis = -1) * (2. / 15.)

    return corr_iso_intra, corr_aniso_intra, corr_iso_inter, corr_aniso_inter

# def calculate_raman(corr: np.ndarray, width: float, dt_ps: float, temperature: float, M: Optional[int] = None):
#     nmax = corr.shape[0] - 1
#     if nmax % 2 != 0:
#         nmax -= 1
#         corr = corr[:-1]
#     tmax = nmax * dt_ps        # ps
#     print('nmax      =', nmax)
#     print('dt   (ps) =', dt_ps)
#     print('tmax (ps) =', tmax)
#     print("width     = ", width)
#     width = width * tmax / 100.0 * 3.0
#     C = apply_gussian_filter(corr, width)
#     freq_ps, CHAT = FT_fft(dt_ps, C, M)
#     d_omega, CHAT = _change_unit(freq_ps, CHAT, temperature)
#     return np.stack([np.arange(CHAT.shape[0]) * d_omega, CHAT], axis = 1)

def calculate_raman(corr: np.ndarray, width: float, dt_ps: float, temperature: float, 
                    M: Optional[int] = None, filter_type: str = "gaussian"):
    nmax = corr.shape[0] - 1
    if nmax % 2 != 0:
        nmax -= 1
        corr = corr[:-1]
    tmax = nmax * dt_ps        # ps
    filter_type = filter_type.lower().strip()
    print('nmax         =', nmax)
    print('dt   (ps)    =', dt_ps)
    print('tmax (ps)    =', tmax)
    print("Filter type  =", filter_type)
    print("width        = ", width)
    if filter_type == "gaussian":
        width = width * tmax / 100.0 * 3
        C = apply_gussian_filter(corr, width)
    elif filter_type == "lorenz":
        C = apply_lorenz_filter(corr, width, dt_ps)
    else:
        raise NotImplementedError(f"Unknown filter type: {filter_type}!")
    freq_ps, CHAT = FT_fft(dt_ps, C, M)
    d_omega, CHAT = _change_unit(freq_ps, CHAT, temperature)
    return np.arange(CHAT.shape[0]) * d_omega, CHAT


def _change_unit(freq_ps, CHAT: np.ndarray, temperature: float):
    cc = 2.99792458e8;                  # m/s
    kB = 1.38064852*1.0e-23             # J/K
    h = 6.62607015e-34                  # J*s
    h_bar = h / (2 * np.pi)
    beta = 1.0 / (kB * temperature);    # J^-1
    freq = 2 * np.pi * freq_ps * 1e12   # s^-1
    CHAT = CHAT * 1e4 * (1 - np.exp(-beta * h_bar * freq * np.arange(CHAT.shape[0])))
    d_omega = 1e10 * freq_ps / cc       # cm^-1
    return d_omega, CHAT