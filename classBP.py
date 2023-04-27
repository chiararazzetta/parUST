# %%
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from f_initialization import (
    grid_create,
    element_discr,
    rit_g,
    sinusoidalPulse,
    resp_probe,
)
from MapGeneration import WideMaps, NarrowMaps
from BpCompute import std_del, NarrowBP, wideMapCut, WideBP, todB


class BeamPattern:

    """Class cpntaining all the methods and the parameters for a Beam Pattern calculation"""

    def __init__(
        self,
        BPtype="Narrow",
        c=1540,
        dt=1e-8,
        min_d=0.002,
        max_d=0.042,
        step=0.25,
        Nel=50,
        Ndepth=400,
        factor=0.5,
        ntimes=1800,
        pad=248,
        pitch=0.245e-3,
        kerf=0.035e-3,
        elevation=5e-3,
        geomf=0.025,
        Nprobe = 192,
        n_xdiscr=40,
        n_ydiscr=150,
        f0=4.5e6,
        Ncycles=3,
        focus=0.025,
        active_el=15,
        wideEl=100,
        probe_respTXT=None,
    ):
        self.BPtype = BPtype

        self.field = {
            "c": c,
            "dt": dt,
            "min_depth": min_d,
            "max_depth": max_d,
            "step": step,
            "Nelfield": Nel,
            "Nz": Ndepth,
            "att_factor": factor,
            "ntimes": ntimes,
            "pad": pad,
        }

        self.probe = {
            "pitch": pitch,
            "kerf": kerf,
            "elevation": elevation,
            "geomf": geomf,
            "N_el": Nprobe,
            "Nx": n_xdiscr,
            "Ny": n_ydiscr,
            "cen": element_discr(pitch, kerf, elevation, n_xdiscr, n_ydiscr),
            "pathProbeResp": probe_respTXT,
            "respProbe": None,
            "idx_freq": None,
        }

        self.probe["lens"] = rit_g(geomf, self.probe["cen"][:, 1], c)

        if probe_respTXT is not None:
            resp, idx = resp_probe(probe_respTXT, ntimes, pad, step)
            self.probe["respProbe"] = resp
            self.probe["idx_freq"] = idx

        gr, idx, nx = grid_create(pitch, Nel, step, Ndepth, min_d, max_d)
        self.field["grid_coord"] = gr
        self.field["Nx"] = nx

        self.pulse = {
            "f0": f0,
            "Ncycles": Ncycles,
            "Pulse": sinusoidalPulse(
                f0, Ncycles, dt, ntimes, pad, self.probe["idx_freq"]
            ),
        }

        self.beam = {
            "H": None,
            "A": None,
            "focus": focus,
            "active_el": active_el,
            "wideH": None,
            "wideEl": wideEl,
            "wideNx": self.field["Nx"],
            "wideNz": self.field["Nz"],
            "wideGrid": self.field["grid_coord"],
        }
        self.beam["delays"] = std_del(
                self.beam["focus"],
                self.probe["pitch"],
                self.field["c"],
                self.beam["active_el"],
            )

    def SaveMaps(self, path):
        if self.BPtype == "Narrow":
            fNarrow = open(path, "wb")
            pickle.dump(
                [
                    self.beam["H"],
                    self.beam["A"],
                    self.field["grid_coord"],
                    self.field["Nx"],
                    self.field["Nz"],
                ],
                fNarrow,
            )
            fNarrow.close()
        elif self.BPtype == "Wide":
            fWide = open(path, "wb")
            pickle.dump(
                [
                    self.beam["H"],
                    self.field["grid_coord"],
                    self.field["Nx"],
                    self.field["Nz"],
                    self.probe["idx_freq"],
                ],
                fWide,
            )
            fWide.close()

    def LoadMaps(self, path):
        if self.BPtype == "Narrow":
            fNarrow = open(path, "rb")
            H, A, g, xnum, znum = pickle.load(fNarrow)
            fNarrow.close()

            self.beam["H"] = H
            self.beam["A"] = A
            self.field["grid_coord"] = g
            self.field["Nx"] = xnum
            self.field["Nz"] = znum
        elif self.BPtype == "Wide":
            fWide = open(path, "rb")
            H, g, xnum, znum, idx = pickle.load(fWide)
            fWide.close()

            self.beam["H"] = H
            self.field["grid_coord"] = g
            self.field["Nx"] = xnum
            self.field["Nz"] = znum
            self.probe["idx_freq"] = idx

    def MapsCompute(self):
        if self.BPtype == "Narrow":
            H, A, g, xnum, znum = NarrowMaps(
                self.probe["pitch"],
                self.probe["cen"],
                self.probe["geomf"],
                self.field["Nelfield"],
                self.field["c"],
                self.field["dt"],
                self.field["step"],
                self.field["Nz"],
                self.field["min_depth"],
                self.field["max_depth"],
                self.field["att_factor"],
                self.pulse["f0"],
            )
            self.beam["H"] = H
            self.beam["A"] = A
            self.field["grid_coord"] = g
            self.field["Nx"] = xnum
            self.field["Nz"] = znum
        elif self.BPtype == "Wide":
            H, xnum, znum, g, idx = WideMaps(
                self.probe["pitch"],
                self.probe["cen"],
                self.probe["geomf"],
                self.probe["N_el"],
                self.field["c"],
                self.field["dt"],
                self.field["step"],
                self.field["Nz"],
                self.field["min_depth"],
                self.field["max_depth"],
                self.field["att_factor"],
                self.field["ntimes"],
                self.field["pad"],
                self.probe["pathProbeResp"],
            )

            self.beam["H"] = H
            self.field["grid_coord"] = g
            self.field["Nx"] = xnum
            self.field["Nz"] = znum
            self.probe["idx_freq"] = idx

    def BPcalculate(self):
        if self.BPtype == "Narrow":
            self.beam["BPlinear"] = NarrowBP(
                self.beam["delays"],
                self.beam["H"],
                self.beam["A"],
                self.pulse["f0"],
                self.beam["active_el"],
            )
            self.beam["BPdecibel"] = todB(self.beam["BPlinear"])
        elif self.BPtype == "Wide":
            maps, xnum, znum, g = wideMapCut(
                self.beam["wideEl"],
                self.field["step"],
                self.beam["H"],
                self.field["Nz"],
                self.field["grid_coord"],
            )
            self.beam["wideH"] = maps
            self.beam["wideNx"] = xnum
            self.beam["wideNz"] = znum
            self.beam["wideGrid"] = g

            self.beam["BPlinear"] = WideBP(
                self.beam["delays"],
                self.beam["wideH"],
                self.beam["active_el"],
                self.field["dt"],
                self.field["ntimes"],
                self.field["pad"],
                self.beam["wideNx"],
                self.beam["wideNz"],
                self.pulse["Pulse"],
                self.probe["idx_freq"],
            )
            self.beam["BPdecibel"] = todB(self.beam["BPlinear"])

    def BPplot(self):
        plt.rcParams["axes.autolimit_mode"] = "round_numbers"
        fig, ax = plt.subplots()
        ax.set_xlabel("Depth (mm)")
        ax.set_ylabel("Probe Line (mm)")
        im = ax.imshow(self.beam["BPdecibel"], cmap="jet")

        ax.set_xticks(np.linspace(0, self.beam["wideNz"], 5))
        xticks = np.round(
            np.linspace(
                np.min(self.beam["wideGrid"][:, 2]),
                np.max(self.beam["wideGrid"][:, 2]),
                5,
            )
            * 1e3,
            1,
        )
        ax.set_xticklabels(xticks)

        ax.set_yticks(np.linspace(0, self.beam["wideNx"], 5))
        yticks = np.round(
            np.linspace(
                np.min(self.beam["wideGrid"][:, 0]),
                np.max(self.beam["wideGrid"][:, 0]),
                5,
            )
            * 1e3,
            1,
        )
        ax.set_yticklabels(yticks)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()
