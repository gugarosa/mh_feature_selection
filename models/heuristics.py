from optimizers import bso, coa
from opytimizer.optimizers.boolean import bmrfo, bpso
from opytimizer.optimizers.evolutionary import de, ga, gp, hs, bsa
from opytimizer.optimizers.misc import doa
from opytimizer.optimizers.population import epo, gwo, hho
from opytimizer.optimizers.science import aso, bh, eo, hgso, mvo, two, wwo
from opytimizer.optimizers.social import qsa, ssd
from opytimizer.optimizers.swarm import (abc, abo, ba, boa, csa, eho, fa, goa,
                                         mfo, pio, pso, sca, sfo, sos, ssa, sso,
                                         woa)


class Heuristic:
    """An Heuristic class helps users in selecting distinct optimization heuristics from the command line.

    """

    def __init__(self, obj, hyperparams):
        """Initialization method.

        Args:
            obj (Optimizer): An Optimizer-child instance.
            hyperparams (dict): Heuristic hyperparams.

        """

        # Creates a property to hold the class itself
        self.obj = obj

        # Creates a property to hold the hyperparams
        self.hyperparams = hyperparams


# Defines an heuristic dictionary constant with the possible values
HEURISTIC = dict(
    abc=Heuristic(abc.ABC, dict(n_trials=10)),
    abo=Heuristic(abo.ABO, dict(sunspot_ratio=0.9, a=2.0)),
    aso=Heuristic(aso.ASO, dict(alpha=50.0, beta=0.2)),
    ba=Heuristic(ba.BA, dict(f_min=0, f_max=2, A=0.5, r=0.5)),
    bh=Heuristic(bh.BH, dict()),
    bmrfo=Heuristic(bmrfo.BMRFO, dict()),
    boa=Heuristic(boa.BOA, dict(c=0.01, a=0.1, p=0.8)),
    bpso=Heuristic(bpso.BPSO, dict()),
    bsa=Heuristic(bsa.BSA, dict(mix_rate=1.0, F=3.0)),
    bso=Heuristic(bso.BSO, dict(k=3, p_one_cluster=0.3, p_one_center=0.4, p_two_centers=0.3)),
    coa=Heuristic(coa.COA, dict(Np=2,Nc=5)),
    csa=Heuristic(csa.CSA, dict(fl=2, AP=0.1)),
    de=Heuristic(de.DE, dict(CR=0.9, F=0.7)),
    doa=Heuristic(doa.DOA, dict(r=1.0)),
    eho=Heuristic(eho.EHO, dict(alpha=0.5, beta=0.1, n_clans=10)),
    eo=Heuristic(eo.EO, dict(a1=2, a2=1, GP=0.5, V=1)),
    epo=Heuristic(epo.EPO, dict(f=2, l=1.5)),
    fa=Heuristic(fa.FA, dict(alpha=0.5, beta=0.2, gamma=1.0)),
    ga=Heuristic(ga.GA, dict(p_selection=0.75, p_mutation=0.25, p_crossover=0.5)),
    gp=Heuristic(gp.GP, dict(p_reproduction=0.25, p_mutation=0.1, p_crossover=0.2, prunning_ratio=0.0)),
    goa=Heuristic(goa.GOA, dict(c_min=0.00001, c_max=1, f=0.5, l=1.5)),
    gwo=Heuristic(gwo.GWO, dict()),
    hgso=Heuristic(hgso.HGSO, dict(n_clusters=2, l1=0.0005, l2=100, l3=0.001, alpha=1.0, beta=1.0, K=1.0)),
    hho=Heuristic(hho.HHO, dict()),
    hs=Heuristic(hs.HS, dict(HMCR=0.7, PAR=0.7, bw=1.0)),
    mfo=Heuristic(mfo.MFO, dict(b=1)),
    mvo=Heuristic(mvo.MVO, dict(WEP_min=0.2, WEP_max=1, p=6)),
    pio=Heuristic(pio.PIO, dict(n_c1=150, n_c2=200, R=0.2)),
    pso=Heuristic(pso.PSO, dict(w=0.7, c1=1.7, c2=1.7)),
    qsa=Heuristic(qsa.QSA, dict()),
    sca=Heuristic(sca.SCA, dict(r_min=0, r_max=2, a=3)),
    sfo=Heuristic(sfo.SFO, dict(PP=0.1, A=4, e=0.001)),
    sos=Heuristic(sos.SOS, dict()),
    ssa=Heuristic(ssa.SSA, dict()),
    ssd=Heuristic(ssd.SSD, dict(c=2.0, decay=0.99)),
    sso=Heuristic(sso.SSO, dict(C_w=0.1, C_p=0.4, C_g=0.9)),
    two=Heuristic(two.TWO, dict(mu_s=1, mu_k=1, delta_t=1, alpha=0.9, beta=0.05)),
    woa=Heuristic(woa.WOA, dict(b=1)),
    wwo=Heuristic(wwo.WWO, dict(h_max=5, alpha=1.001, beta=0.001, k_max=1))
)

def get_heuristic(name):
    """Gets an heuristic by its identifier.

    Args:
        name (str): Heuristic's identifier.

    Returns:
        An instance of the MetaHeuristic class.

    """

    # Tries to invoke the method
    try:
        # Returns the corresponding object
        return HEURISTIC[name]

    # If object is not found
    except:
        # Raises a RuntimeError
        raise RuntimeError(f'Heuristic {name} has not been specified yet.')
