import numpy as np
import pandas as pd
import utils
import gzip
import os
import subprocess
import tempfile
from rdkit import Chem

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
            if potential_no_absoroption == True:
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


def fitness_function(population, checkpoint_dirs, unit_list, ref_features_row, ref_features_cols, maximize):
    """
    Score a population using an ensemble of ChemProp v1 models via the CLI.
    Each molecule is evaluated under the fixed reference formulation conditions.
    Returns [ranked_scores, ranked_population, ranked_stds] ordered best-first,
    where "best" is the highest score when maximize is True and the lowest when
    it is False. The std is the spread across the ensemble for that molecule:
    a large value means the models disagree about it.
    """
    smiles_list = [Chem.MolToSmiles(utils.make_molecule(p, unit_list)) for p in population]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Population SMILES — column name must match what the models were trained on
        smiles_path = os.path.join(tmpdir, "population.csv")
        pd.DataFrame({"IL_SMILES": smiles_list}).to_csv(smiles_path, index=False)

        # Reference formulation features — one row per molecule, same values for all
        features_path = os.path.join(tmpdir, "features.csv")
        pd.DataFrame(
            [ref_features_row] * len(smiles_list), columns=ref_features_cols
        ).to_csv(features_path, index=False)

        all_preds = []
        for i, ckpt_dir in enumerate(checkpoint_dirs):
            print(f"  Scoring with model {i+1}/{len(checkpoint_dirs)}...", flush=True)
            preds_path = os.path.join(tmpdir, f"preds_{i}.csv")
            result = subprocess.run(
                [
                    "chemprop_predict",
                    "--checkpoint_dir", ckpt_dir,
                    "--test_path", smiles_path,
                    "--features_path", features_path,
                    "--preds_path", preds_path,
                    # chemprop's default of 8 DataLoader workers deadlocks on macOS
                    "--num_workers", "0",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"chemprop_predict failed for {ckpt_dir}:\n{result.stderr}"
                )
            all_preds.append(pd.read_csv(preds_path)["Experiment_value"].tolist())

    all_preds = np.array(all_preds)
    score_list = list(all_preds.mean(axis=0))
    std_list = list(all_preds.std(axis=0))

    ranked_indices = list(np.argsort(score_list))
    if maximize:
        ranked_indices.reverse()
    ranked_score = [score_list[i] for i in ranked_indices]
    ranked_pop = [population[i] for i in ranked_indices]
    ranked_std = [std_list[i] for i in ranked_indices]
    return [ranked_score, ranked_pop, ranked_std]


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
            "/ihome/ghutchison/blp62/GA_best_practices/Calculations/GFN2/"
            + filename
            + ".out"
        )
        GFN2_props = parse_GFN2(GFN2_file)  # dipole_moment, polarizability
        polarizability = GFN2_props[1]
        return polarizability

    elif scoring_prop == "opt_bg":
        stda_file = (
            "/ihome/ghutchison/blp62/GA_best_practices/Calculations/sTDDFTxtb/"
            + filename
            + ".stda"
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
        except:
            ratio_water_hexane = 100000

        return ratio_water_hexane
    else:
        print("Not a valid scoring property")
        return None
