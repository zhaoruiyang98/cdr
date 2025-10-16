from __future__ import annotations
from cosmoprimo import Cosmology
from cosmoprimo.fiducial import DESI as DESI


def Quijote(extra_params=None, **kwargs):
    cosmo = Cosmology(
        engine="camb",
        omega_b=0.02206838529,
        omega_cdm=0.120925743885,
        A_s=2.13e-9,
        n_s=0.9624,
        h=0.6711,
        m_ncdm=[],
        N_eff=3.046,
        T_cmb=2.725,
        YHe=0.24,
        tau_reio=0.0925,
        extra_params=extra_params,
        **kwargs,
    )
    cosmo.engine._camb_params.WantScalars = False
    cosmo.engine._camb_params.Transfer.k_per_logint = 50
    cosmo.engine._camb_params.YHe = 0.24
    return cosmo
