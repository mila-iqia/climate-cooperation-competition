"""PLS climate club: probabilistic logic shields as soft club compliance on JAX RICE-N.

Shield reference: Yang, Marra, Rens & De Raedt (2023), "Safe Reinforcement
Learning via Probabilistic Logic Shields", IJCAI.
Club reference: Nordhaus (2015, AER 105(4)), "Climate Clubs".
"""

from .env import PLSClubMediator, PLSNaiveClubRice
from .shield import ShieldedCategoricalLayer, mitigation_safety_weights

__all__ = [
    "PLSClubMediator",
    "PLSNaiveClubRice",
    "ShieldedCategoricalLayer",
    "mitigation_safety_weights",
]
