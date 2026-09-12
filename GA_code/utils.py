import pandas as pd
from ugi_reaction import get_ugi_product

# def make_unit_list():
#     '''
#     Makes a dataframe containing the monomer SMILES
#     Returns
#     -------
#     units: dataframe
#         contains 1 column of monomer SMILES
#     '''
#     units = pd.read_csv('../monomer_SMILES.csv', index_col=False)

#     return units


def make_unit_list(aldehydes, amines, isocyanides, acids=None):
    """
    Load the building-block libraries.

    Key order defines the genome layout: index i of an individual selects from
    the i-th library here. Omitting acids yields a three-slot genome for the
    three-component reaction, and everything downstream sizes itself off this
    dict rather than assuming four.
    """

    def load(path):
        return pd.read_csv(path, usecols=["smiles"], index_col=False)

    units = {"aldehyde": load(aldehydes)}
    if acids is not None:
        units["acid"] = load(acids)
    units["amine"] = load(amines)
    units["isocyanide"] = load(isocyanides)
    return units


def make_file_name(polymer):
    """
    Makes file name for a given polymer

    Parameters
    ---------
    polymer: list (specific format)
        [A, B]

    Returns
    -------
    file_name: str
        polymer file name (w/o extension) showing monomer indicies and full sequence
        e.g. 100_200_101010 for a certain hexamer
    """

    # capture monomer indexes as strings for file naming
    mono1 = str(polymer[0])
    mono2 = str(polymer[1])

    # make file name string
    file_name = "%s_%s_010101" % (mono1, mono2)

    return file_name


def make_molecule(polymer, unit_list):
    smiles = {comp: unit_list[comp].iloc[idx, 0] for comp, idx in zip(unit_list, polymer)}
    return get_ugi_product(
        primary_amine=smiles["amine"],
        aldehyde=smiles["aldehyde"],
        isocyanide=smiles["isocyanide"],
        # absent for the three-component reaction
        carboxylic_acid=smiles.get("acid"),
    )


def binSearch(wheel, num):
    """
    Finds what pie (or individual) in a wheel the number belongs in. Works with the SUS selection method

    Parameters
    ----------
    wheel: list
        contains list of lists of format [lower_limit, upper_limit, ranked_scores, ranked_population]
    num: float
        random number between 0 and 1

    Returns
    -------
    score: float
        fitness score of bin
    polymer: list
        [mon_1_index, mon_2_index]
    """
    mid = len(wheel) // 2
    low, high, score, polymer = wheel[mid]
    if low <= num <= high:
        return score, polymer
    elif high < num:
        return binSearch(wheel[mid + 1 :], num)
    else:
        return binSearch(wheel[:mid], num)


def rank_binSearch(wheel, num):
    """
    Finds what pie in a wheel the number belongs in. Works with the rank selection method

    Parameters
    ----------
    wheel: list
        contains list of lists of format [lower_limit, upper_limit, ranked_scores, ranked_population]
    num: float
        random number between 0 and 1

    Returns
    -------
    score: float
        fitness score of bin
    polymer: list
        [mon_1_index, mon_2_index]
    """
    mid = len(wheel) // 2
    low, high, polymer = wheel[mid]
    if low <= num <= high:
        return polymer
    elif high < num:
        return rank_binSearch(wheel[mid + 1 :], num)
    else:
        return rank_binSearch(wheel[:mid], num)


# Building blocks are identified across all four slots in one integer namespace.
# The stride exceeds every library size, so an aldehyde index cannot collide with
# an acid index carrying the same number.
_SLOT_STRIDE = 10_000_000


def update_block_freq(population, freq):
    """
    Accumulate how often each building block has been used, over the whole run.

    freq maps block id -> times used. Only blocks actually drawn are stored, so
    this stays small even though the libraries hold hundreds of thousands of rows.
    """
    for genome in population:
        for slot, idx in enumerate(genome):
            uid = slot * _SLOT_STRIDE + idx
            freq[uid] = freq.get(uid, 0) + 1
    return freq


def blocks_used_per_slot(freq):
    """
    How many distinct building blocks each genome slot has drawn so far,
    keyed by slot index (0 carbonyl, 1 acid, 2 amine, 3 isocyanide).
    """
    counts = {}
    for uid in freq:
        slot = uid // _SLOT_STRIDE
        counts[slot] = counts.get(slot, 0) + 1
    return counts


def top_ranked_blocks(freq, n=10, min_count=2):
    """
    The n most-used building blocks, most-used first.

    Blocks drawn only once are ignored. With libraries this large most blocks are
    singletons, and including them would make the leaderboard a tie broken on id
    alone -- which looks perfectly stable while the population is in fact still
    churning, reporting convergence that has not happened.

    Ties among the survivors break on block id so the ordering is reproducible
    between generations; otherwise equally-used blocks could swap places and look
    like real churn.
    """
    qualifying = ((uid, c) for uid, c in freq.items() if c >= min_count)
    ordered = sorted(qualifying, key=lambda kv: (-kv[1], kv[0]))
    return [uid for uid, _ in ordered[:n]]


def not_valid(temp_child, unit_list):
    try:
        mol = make_molecule(temp_child, unit_list)
        return mol is None
    except Exception:
        return True
