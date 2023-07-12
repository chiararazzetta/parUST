# %%
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import cupy as cp

from source.f_initialization import (
    grid_create,
    element_discr,
    rit_g,
    sinusoidalPulse,
    resp_probe,
)
from source.MapGeneration import WideMaps, NarrowMaps
from source.BpCompute import std_del, NarrowBP, WideBP, todB


class BP_symm:

    """Class containing all the methods and the parameters for symmetric BP calculation"""

    def __init__(
        self,
        BPtype="Narrow",
        c=1540,
        dt=1e-8,
        min_d=0.002,
        max_d=0.042,
        step=0.25,
        Ndepth=400,
        factor=0.5,
        ntimes=1800,
        pad=248,
        pitch=0.245e-3,
        kerf=0.035e-3,
        elevation=5e-3,
        geomf=0.025,
        Nprobe=192,
        n_xdiscr=40,
        n_ydiscr=150,
        f0=4.5e6,
        Ncycles=3,
        focus=[0.025],
        active_el=[4],
        NelImm=70,
        probe_respTXT=None,
        apo=0,
        sigma=[1.5],
        APOtype=["gauss"],
        device="cpu"
    ):
        self.BPtype = BPtype
        self.device = device

        self.field = {
            "c": c,
            "dt": dt,
            "min_depth": min_d,
            "max_depth": max_d,
            "step": step,
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

        gr, idx, nx = grid_create(pitch, Nprobe, step, Ndepth, min_d, max_d)
        self.field["grid_coord"] = gr
        self.field["Nx"] = nx

        self.pulse = {
            "f0": f0,
            "Ncycles": Ncycles,
            "Pulse": sinusoidalPulse(
                f0, Ncycles, dt, ntimes, pad, self.probe["idx_freq"], device
            )
        }

        self.beam = {
            "H": None,
            "A": None,
            "focus": list(focus),
            "active_el": list(active_el),
            "Nsets": len(focus) * len(active_el),
            "wideH": None,
            "NelImm": NelImm,
            "apo": apo,
            "sigma": sigma,
            "APOtype": APOtype,
            "NX": list(),
            "NZ": list(),
            "BPgrid": list(),
            "dbCut": -40,
            "delays": list(),
            "BPlinear": list(),
            "BPdecibel": list()
        }
        self.beam["delays"].append(std_del(
            self.beam["focus"][0],
            self.probe["pitch"],
            self.field["c"],
            self.beam["active_el"][0],
            self.device
        ))

    def DelaysSet(self, free_del=None):
        if free_del is None:
            self.beam["Nsets"] = len(
                self.beam["focus"]) * len(self.beam["active_el"])
            self.beam["delays"] = []
            for i in range(len(self.beam["active_el"])):
                for j in range(len(self.beam["focus"])):
                    self.beam["delays"].append(std_del(
                        self.beam["focus"][j],
                        self.probe["pitch"],
                        self.field["c"],
                        self.beam["active_el"][i],
                        self.device
                    ))
        else:
            self.beam["delays"] = free_del
            self.beam["Nsets"] = len(free_del)
            self.beam["focus"] = None

        if self.beam["apo"] == 1 and self.beam["Nsets"] != len(self.beam["sigma"]):
            raise Exception(
                "The number of delay curves differs from the number of apodization curves")

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

            self.beam["H"] = np.asarray(H)
            self.beam["A"] = np.asarray(A)
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
                self.field["c"],
                self.field["dt"],
                self.probe["geomf"],
                self.field["step"],
                self.field["Nz"],
                self.field["min_depth"],
                self.field["max_depth"],
                self.field["att_factor"],
                self.probe["cen"],
                self.pulse["f0"],
                self.probe["N_el"],
                self.beam["NelImm"]
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
                self.probe["pathProbeResp"]
            )

            self.beam["H"] = H
            self.field["grid_coord"] = g
            self.field["Nx"] = xnum
            self.field["Nz"] = znum
            self.probe["idx_freq"] = idx

    def BPcompute(self):
        if self.BPtype == "Narrow":
            if self.beam["apo"] == 0:
                for i in range(self.beam["Nsets"]):
                    B = NarrowBP(
                        self.beam["delays"][i],
                        self.beam["H"],
                        self.beam["A"],
                        self.pulse["f0"],
                        self.beam["delays"][i].shape[0],
                        self.beam["apo"],
                        self.beam["sigma"][0],
                        self.beam["APOtype"][0],
                        self.device
                    )
                    self.beam["BPlinear"].append(B)
                    self.beam["BPgrid"].append(self.field["grid_coord"])
                    self.beam["NX"].append(self.field["Nx"])
                    self.beam["NZ"].append(self.field["Nz"])
                    self.beam["BPdecibel"].append(
                        todB(B, self.beam["dbCut"], self.device))
            elif self.beam["apo"] == 1:
                for i in range(self.beam["Nsets"]):
                    B = NarrowBP(
                        self.beam["delays"][i],
                        self.beam["H"],
                        self.beam["A"],
                        self.pulse["f0"],
                        self.beam["delays"][i].shape[0],
                        self.beam["apo"],
                        self.beam["sigma"][i],
                        self.beam["APOtype"][i],
                        self.device
                    )
                    self.beam["BPlinear"].append(B)
                    self.beam["BPgrid"].append(self.field["grid_coord"])
                    self.beam["NX"].append(self.field["Nx"])
                    self.beam["NZ"].append(self.field["Nz"])
                    self.beam["BPdecibel"].append(
                        todB(B, self.beam["dbCut"], self.device))
        elif self.BPtype == "Wide":
            if self.beam["apo"] == 0:
                for i in range(self.beam["Nsets"]):
                    B, xnum, znum, g = WideBP(
                        self.beam["delays"][i],
                        self.beam["H"],
                        self.beam["delays"][i].shape[0],
                        self.field["step"],
                        self.beam["NelImm"],
                        self.field["grid_coord"],
                        self.field["dt"],
                        self.field["ntimes"],
                        self.field["pad"],
                        self.field["Nz"],
                        self.pulse["Pulse"],
                        self.probe["idx_freq"],
                        self.beam["apo"],
                        self.beam["sigma"][0],
                        self.beam["APOtype"][0],
                        self.device

                    )
                    self.beam["BPlinear"].append(B)
                    self.beam["BPgrid"].append(g)
                    self.beam["NX"].append(xnum)
                    self.beam["NZ"].append(znum)
                    self.beam["BPdecibel"].append(
                        todB(B, self.beam["dbCut"], self.device))
            elif self.beam["apo"] == 1:
                for i in range(self.beam["Nsets"]):
                    B, xnum, znum, g = WideBP(
                        self.beam["delays"][i],
                        self.beam["H"],
                        self.beam["delays"][i].shape[0],
                        self.field["step"],
                        self.beam["NelImm"],
                        self.field["grid_coord"],
                        self.field["dt"],
                        self.field["ntimes"],
                        self.field["pad"],
                        self.field["Nz"],
                        self.pulse["Pulse"],
                        self.probe["idx_freq"],
                        self.beam["apo"],
                        self.beam["sigma"][0],
                        self.beam["APOtype"][0],
                        self.device

                    )
                    self.beam["BPlinear"].append(B)
                    self.beam["BPgrid"].append(g)
                    self.beam["NX"].append(xnum)
                    self.beam["NZ"].append(znum)
                    self.beam["BPdecibel"].append(
                        todB(B, self.beam["dbCut"], self.device))

    def BPplot(self, Nfig=0):
        if self.device == "cpu":
            BP = self.beam["BPdecibel"][Nfig]
            N = self.beam["NZ"][Nfig]
            G = self.beam["BPgrid"][Nfig]
            X = self.beam["NX"][Nfig]
        elif self.device == "gpu":
            BP = cp.asnumpy(self.beam["BPdecibel"][Nfig])
            N = cp.asnumpy(self.beam["NZ"][Nfig])
            G = cp.asnumpy(self.beam["BPgrid"][Nfig])
            X = cp.asnumpy(self.beam["NX"][Nfig])

        plt.rcParams["axes.autolimit_mode"] = "round_numbers"
        fig, ax = plt.subplots()
        ax.set_xlabel("Depth (mm)")
        ax.set_ylabel("Probe Line (mm)")
        im = ax.imshow(BP, cmap="jet")

        ax.set_xticks(np.linspace(0, N, 5))
        xticks = np.round(
            np.linspace(
                np.min(G[:, 2]),
                np.max(G[:, 2]),
                5,
            )
            * 1e3,
            1,
        )
        ax.set_xticklabels(xticks)

        ax.set_yticks(np.linspace(0, X, 5))
        yticks = np.round(
            np.linspace(
                np.min(G[:, 0]),
                np.max(G[:, 0]),
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
