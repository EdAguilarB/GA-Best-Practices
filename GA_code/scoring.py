import gzip
import os
import subprocess
import tempfile
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import utils
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

# ECFP4. Built once because constructing a generator per call is wasteful when
# every candidate in every generation is fingerprinted the same way.
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Everything the diversity objective needs: the reference pool to measure
# novelty against, the bounds that put predictions on the reference's scale,
# and how much of the combined score dissimilarity accounts for.
DiversityRef = namedtuple("DiversityRef", "fingerprints pred_min pred_max weight norm_mode")


def normalise_predictions(predictions, reference):
    """
    Put predictions on [0, 1] so they can be blended with dissimilarity.

    "reference" scales against the bounds of the reference data, which keeps a
    value comparable across runs but compresses hard: ensemble means regress
    toward the mean and occupy a fraction of the range the raw data spans.

    "population" min-maxes within the batch being scored, so the spread always
    fills [0, 1] and the two objectives carry comparable weight. The cost is
    that a value only means something relative to the batch it came from, which
    is why the summary recomputes it across every candidate at the end.

    "none" blends the raw prediction straight into the objective. Neither term
    is rescaled, so each contributes in proportion to how much it actually
    varies rather than to a range imposed on it -- at the price of the
    prediction being unbounded and free to leave the [0, 1] dissimilarity sits in.
    """
    if reference.norm_mode == "none":
        return list(predictions)

    if reference.norm_mode == "population":
        low, high = min(predictions), max(predictions)
    else:
        low, high = reference.pred_min, reference.pred_max

    span = high - low
    if span <= 0:
        # every candidate identical, so nothing to separate them by
        return [0.5] * len(predictions)
    return [min(1.0, max(0.0, (p - low) / span)) for p in predictions]


def load_diversity_reference(reference_path, weight, norm_mode="reference"):
    """
    Build the reference pool that candidates are measured against for novelty.

    Duplicate SMILES are collapsed: a molecule appearing twice cannot change any
    candidate's nearest neighbour, and the pool here is roughly half duplicates,
    so dropping them halves the similarity work every generation.

    Prediction bounds come from the reference's own Experiment_value_normalized
    column, which puts the model's output on the scale of the data it was
    trained against.
    """
    df = pd.read_csv(reference_path, usecols=["smiles", "Experiment_value_normalized"])

    smiles = df["smiles"].dropna().unique()
    fingerprints = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fingerprints.append(_MORGAN.GetFingerprint(mol))

    bounds = df["Experiment_value_normalized"].dropna()
    return DiversityRef(fingerprints, float(bounds.min()), float(bounds.max()), weight, norm_mode)


def nearest_neighbour_dissimilarity(mols, reference):
    """
    1 - the Tanimoto similarity to the closest molecule in the reference pool.

    0 means the candidate is already in the pool; 1 means it shares no
    fingerprint bits with anything there.
    """
    scores = []
    for mol in mols:
        fp = _MORGAN.GetFingerprint(mol)
        sims = DataStructs.BulkTanimotoSimilarity(fp, reference.fingerprints)
        scores.append(1.0 - max(sims))
    return scores


def parse_GFN2(filename):
    """
    Parses through GFN2-xTB output files

    Parameters
    -----------
    filename: str
        path to output file

    Returns
    -------
    outputs: list
        [dipole_moment, polarizability]
    """

    with open(filename, "r", encoding="utf-8") as file:
        line = file.readline()
        while line:
            if "molecular dipole" in line:
                line = file.readline()
                line = file.readline()
                line = file.readline()

                line_list = line.split()
                dipole_moment = float(line_list[-1])

            elif "Mol. C8AA" in line:
                line = file.readline()
                line_list = line.split()

                polarizability = float(line_list[-1])

            line = file.readline()
        line = file.readline()

        outputs = [dipole_moment, polarizability]

        return outputs


def parse_GFN2_gzip(filename):
    """
    Parses through gzipped GFN2-xTB output files

    Parameters
    -----------
    filename: str
        path to output file

    Returns
    -------
    outputs: list
        [dipole_moment, polarizability]
    """

    with gzip.open(filename, "rt") as file:
        line = file.readline()
        while line:
            if "molecular dipole" in line:
                line = file.readline()
                line = file.readline()
                line = file.readline()

                line_list = line.split()
                dipole_moment = float(line_list[-1])

            elif "Mol. C8AA" in line:
                line = file.readline()
                line_list = line.split()

                polarizability = float(line_list[-1])

            line = file.readline()
        line = file.readline()

        outputs = [dipole_moment, polarizability]

        return outputs


def parse_sTDA(filename):
    """
    Parses through sTD-DFT-xTB output files

    Parameters
    -----------
    filename: str
        path to output file

    Returns
    -------
    opt_bg: float
        optical bandgap
        energy of the first transition within the first 12 transition with an oscillator strength greater than 0.5
    """
    with open(filename, "r", encoding="utf-8") as file:
        line = file.readline()
        oscs = []
        energyEV = []
        potential_no_absoroption = False
        while line:
            if "excitation energy, transition moments and TDA amplitudes" in line:
                line = file.readline()
                line = file.readline()
                line = file.readline()
                line_list = line.split()
                while line != "\n":
                    line_list = line.split()
                    oscs.append(float(line_list[3]))
                    energyEV.append(float(line_list[1]))
                    line = file.readline()
            elif "excitation energies, transition moments and TDA amplitudes" in line:
                line = file.readline()
                line = file.readline()
                line_list = line.split()
                while line != "\n":
                    line_list = line.split()
                    oscs.append(float(line_list[3]))
                    energyEV.append(float(line_list[1]))
                    line = file.readline()

            elif "0 CSF included by energy" in line:
                potential_no_absoroption = True

            line = file.readline()
        line = file.readline()

        if len(oscs) != 0:
            opt_bg = round(energyEV[0], 4)

            # Opt bg is the energy of the first transition within the first 12 transition with an oscillator strength greater than 0.5
            if len(oscs) < 12:
                for i in range(len(oscs)):
                    if oscs[i] > 0.5:
                        opt_bg = round(energyEV[i], 4)
                        break
            else:
                for x in range(12):
                    if oscs[x] > 0.5:
                        opt_bg = round(energyEV[x], 4)
                        break

            return opt_bg
        else:
            if potential_no_absoroption:
                print("no absorption below 5 eV")
            else:
                print("error with file")
            print(filename)
            opt_bg = 10
            return opt_bg


def solvation(filename):
    """
    Parses through xTB output files for solvation energy

    Parameters
    -----------
    filename: str
        path to output file

    Returns
    -------
    solvation_energy: float
        solvation energy
    """
    with open(filename, "r", encoding="utf-8") as file:
        line = file.readline()
        while line:
            if "-> Gsolv" in line:
                line_list = line.split()
                solvation_energy = float(line_list[3])
                break

            line = file.readline()
        line = file.readline()

    return solvation_energy


def compute_reference_row(extra_x_path):
    """
    Compute per-column means across all training data.
    Returns (row_values, col_names) to use as a fixed reference condition
    when scoring new molecules whose non-SMILES formulation features are unknown.
    """
    df = pd.read_csv(extra_x_path)
    return df.mean().tolist(), list(df.columns)


def fitness_function(
    population,
    checkpoint_dirs,
    unit_list,
    ref_features_row,
    ref_features_cols,
    maximize,
    n_jobs=1,
    diversity=None,
):
    """
    Score a population using an ensemble of ChemProp v1 models via the CLI.
    Each molecule is evaluated under the fixed reference formulation conditions.

    Returns a list ordered best-first, where "best" is the highest score when
    maximize is True and the lowest when it is False:

        [ranked_objective, population, std, prediction, dissimilarity, norm_prediction]

    Without a diversity reference the objective is the raw ensemble mean and the
    last two entries mirror it. With one, the objective becomes a weighted blend
    of the normalised prediction and the candidate's distance from the reference
    pool, while prediction and std stay untransformed so the raw model output is
    never lost.
    """
    mols = [utils.make_molecule(p, unit_list) for p in population]
    smiles_list = [Chem.MolToSmiles(m) for m in mols]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Population SMILES — column name must match what the models were trained on
        smiles_path = os.path.join(tmpdir, "population.csv")
        pd.DataFrame({"IL_SMILES": smiles_list}).to_csv(smiles_path, index=False)

        # Reference formulation features — one row per molecule, same values for all
        features_path = os.path.join(tmpdir, "features.csv")
        pd.DataFrame([ref_features_row] * len(smiles_list), columns=ref_features_cols).to_csv(
            features_path, index=False
        )

        def predict_with(job):
            i, ckpt_dir = job
            preds_path = os.path.join(tmpdir, f"preds_{i}.csv")
            result = subprocess.run(
                [
                    "chemprop_predict",
                    "--checkpoint_dir",
                    ckpt_dir,
                    "--test_path",
                    smiles_path,
                    "--features_path",
                    features_path,
                    "--preds_path",
                    preds_path,
                    # chemprop's default of 8 DataLoader workers deadlocks on macOS
                    "--num_workers",
                    "0",
                ],
                capture_output=True,
                text=True,
                # each model scores the same handful of molecules, so the work per
                # process is trivial; letting each spawn its own thread pool just
                # makes them fight over cores while running concurrently
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            )
            if result.returncode != 0:
                raise RuntimeError(f"chemprop_predict failed for {ckpt_dir}:\n{result.stderr}")
            return pd.read_csv(preds_path)["Experiment_value"].tolist()

        # The ensemble members are independent and each spends most of its time
        # in interpreter start-up rather than computing, so running them at once
        # collapses roughly ten start-ups into one. Threads suffice: every worker
        # is blocked in subprocess.run, holding no GIL.
        workers = min(n_jobs, len(checkpoint_dirs))
        print(
            f"  scoring {len(smiles_list)} molecules against "
            f"{len(checkpoint_dirs)} models ({workers} at a time)...",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            all_preds = list(pool.map(predict_with, enumerate(checkpoint_dirs)))

    all_preds = np.array(all_preds)
    pred_list = list(all_preds.mean(axis=0))
    std_list = list(all_preds.std(axis=0))

    if diversity is None:
        objective = pred_list
        dissim_list = pred_list
        norm_pred_list = pred_list
    else:
        dissim_list = nearest_neighbour_dissimilarity(mols, diversity)
        norm_pred_list = normalise_predictions(pred_list, diversity)
        w = diversity.weight
        objective = [(1.0 - w) * n + w * d for n, d in zip(norm_pred_list, dissim_list)]

    ranked_indices = list(np.argsort(objective))
    if maximize:
        ranked_indices.reverse()

    def in_rank_order(values):
        return [values[i] for i in ranked_indices]

    return [
        in_rank_order(objective),
        in_rank_order(population),
        in_rank_order(std_list),
        in_rank_order(pred_list),
        in_rank_order(dissim_list),
        in_rank_order(norm_pred_list),
    ]


def fitness_individual(polymer, scoring_prop):
    """
    Returns the fitness score of an individual

    Parameters
    ----------
    polymer: list
        specific order [monomer 1 index, monomer 2 index]
    scoring_prop: str
        can be 'polar', 'opt_bg', or 'solv_eng'

    Returns
    -------
    Returns the score depending on the property (polarizability, optical bandgap, or solvation ratio)
    """

    filename = utils.make_file_name(polymer)
    if scoring_prop == "polar":
        GFN2_file = (
            "/ihome/ghutchison/blp62/GA_best_practices/Calculations/GFN2/" + filename + ".out"
        )
        GFN2_props = parse_GFN2(GFN2_file)  # dipole_moment, polarizability
        polarizability = GFN2_props[1]
        return polarizability

    elif scoring_prop == "opt_bg":
        stda_file = (
            "/ihome/ghutchison/blp62/GA_best_practices/Calculations/sTDDFTxtb/" + filename + ".stda"
        )
        opt_bg = parse_sTDA(stda_file)
        return opt_bg

    elif scoring_prop == "solv_eng":
        solv_water_file = (
            "/ihome/ghutchison/blp62/GA_best_practices/Calculations/solvation_water/"
            + filename
            + ".out"
        )
        solv_hexane_file = (
            "/ihome/ghutchison/blp62/GA_best_practices/Calculations/solvation_hexane/"
            + filename
            + ".out"
        )

        try:
            # calculate solvation free energy of acceptor in water
            solv_water = solvation(solv_water_file)
            # calculate solvation free energy of acceptor in hexane
            solv_hexane = solvation(solv_hexane_file)

            # ratio of water solvation energy to hexane solvation energy
            ratio_water_hexane = (solv_water - solv_hexane) / abs(solv_water)
        except Exception:
            ratio_water_hexane = 100000

        return ratio_water_hexane
    else:
        print("Not a valid scoring property")
        return None
