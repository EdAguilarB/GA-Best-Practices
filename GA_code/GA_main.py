# Imports
import argparse
import ast
import csv
import os
import pickle
import random
import time
from copy import deepcopy

import numpy as np
import scoring
import utils
from rdkit import Chem
from scipy.stats import spearmanr

# Genome layout: index i of an individual selects from COMPONENT_SLOTS[i]. Coverage
# reporting decodes block ids by the same order, so keep it the single definition.
COMPONENT_SLOTS = ["aldehyde", "acid", "amine", "isocyanide"]


def main(
    chem_property,
    run_label,
    checkpoint_dirs,
    features_ref_path,
    n_generations=500,
    aldehydes=None,
    acids=None,
    amines=None,
    isocyanides=None,
    maximize=False,
    output_dir=".",
    pop_size=32,
    selection_method="tournament_3",
    mutation_rate=0.4,
    elitism_perc=0.5,
    spear_thresh=0.8,
    conv_gen=50,
    n_jobs=1,
):
    # reuse initial state — set to "y" to replay from a saved randstate file
    initial_restart = "n"
    # flag for restart from save
    restart = "n"
    # Run number (for use with same initial states), can be A, B, C, D, or E
    # run_label = 'D'

    # GA run file name, encoding the hyperparameters actually used
    run_name = "%s_%s_%s" % (selection_method, chem_property, run_label)

    # Pre-compute reference formulation features once (mean of each column across all training data)
    print("Loading reference features...", flush=True)
    ref_features_row, ref_features_cols = scoring.compute_reference_row(features_ref_path)

    # Create list of possible building block unit SMILES in specific format
    print("Loading component CSVs...", flush=True)
    unit_list = utils.make_unit_list(aldehydes, acids, amines, isocyanides)
    print("Done loading. Starting GA...", flush=True)

    # every artifact from this run lands here
    os.makedirs(output_dir, exist_ok=True)
    last_gen_path = os.path.join(output_dir, "last_gen_" + run_name + ".p")
    randstate_path = os.path.join(output_dir, "randstate_" + run_name + ".p")
    initial_randstate = os.path.join(output_dir, "initial_randstate_" + run_label + ".p")

    if restart == "y":
        # reload parameters and random state from restart file
        last_gen_filename = last_gen_path
        open_params = open(last_gen_filename, "rb")
        params = pickle.load(open_params)
        open_params.close()

        # inject runtime values (not stored in pickle; always loaded fresh from CLI args)
        while len(params) < 19:
            params.append(None)
        params[9] = checkpoint_dirs
        params[10] = ref_features_row
        params[11] = ref_features_cols
        params[12] = maximize
        params[13] = output_dir
        params[16] = spear_thresh
        params[18] = n_jobs

        randstate_filename = randstate_path
        open_rand = open(randstate_filename, "rb")
        randstate = pickle.load(open_rand)
        random.setstate(randstate)
        open_rand.close()

    else:
        if initial_restart == "n":
            # sets initial state
            randstate = random.getstate()
            rand_file = open(initial_randstate, "wb")
            pickle.dump(randstate, rand_file)
            rand_file.close()
        else:
            # re-opens saved initial state for exact reproducibility
            open_rand = open(initial_randstate, "rb")
            randstate = pickle.load(open_rand)
            random.setstate(randstate)
            open_rand.close()

        # run initial generation if NOT loading from restart file
        params = init_gen(
            pop_size,
            selection_method,
            mutation_rate,
            elitism_perc,
            run_name,
            chem_property,
            unit_list,
            checkpoint_dirs,
            ref_features_row,
            ref_features_cols,
            maximize,
            output_dir,
            spear_thresh,
            n_jobs,
        )

        # pickle parameters needed for restart
        params_file = open(last_gen_path, "wb")
        pickle.dump(params, params_file)
        params_file.close()

        # pickle random state for restart
        randstate = random.getstate()
        rand_file = open(randstate_path, "wb")
        pickle.dump(randstate, rand_file)
        rand_file.close()

    started = time.time()
    for completed in range(1, n_generations + 1):
        # run next generation of GA
        params = next_gen(params)

        # pickle parameters needed for restart
        params_file = open(last_gen_path, "wb")
        pickle.dump(params, params_file)
        params_file.close()

        # pickle random state for restart
        randstate = random.getstate()
        rand_file = open(randstate_path, "wb")
        pickle.dump(randstate, rand_file)
        rand_file.close()

        # progress: the run ends at whichever comes first, the generation cap or
        # convergence, so report position against both.
        scores = params[3][0]
        per_gen = (time.time() - started) / completed
        eta = per_gen * (n_generations - completed)
        print(
            f"[gen {params[2]:>4}/{n_generations + 1}] "
            f"best {scores[0]:>8.4f}  med {scores[len(scores) // 2]:>8.4f}  |  "
            f"spearman {params[17]:>5.2f}  converge {params[15]:>3}/{conv_gen}  |  "
            f"{per_gen / 60:.1f} min/gen  elapsed {(time.time() - started) / 60:.0f}m  "
            f"cap eta {eta / 60:.0f}m",
            flush=True,
        )

        if params[15] >= conv_gen:
            print(
                f"Converged at generation {params[2]}: Spearman > {spear_thresh} "
                f"for {conv_gen} consecutive generations",
                flush=True,
            )
            break

    summary_path, n_unique = write_summary(output_dir, run_name, unit_list, maximize)
    print(f"Wrote {n_unique} unique candidates to {summary_path}", flush=True)
    report_coverage(unit_list, params[14], n_unique)


def report_coverage(unit_list, block_freq, n_products):
    """
    Log how much of the search space the run actually touched.

    Per component this is the share of that library the GA has drawn at least
    once. For products it is unique Ugi products scored against every
    combination the four libraries admit -- a number so large that the share is
    best read as "effectively none of it", which is the honest picture of what
    any GA covers here.
    """
    used = utils.blocks_used_per_slot(block_freq)

    total_products = 1
    for comp in COMPONENT_SLOTS:
        total_products *= len(unit_list[comp])

    print("\nSearch space covered", flush=True)
    for slot, comp in enumerate(COMPONENT_SLOTS):
        library = len(unit_list[comp])
        tried = used.get(slot, 0)
        label = "carbonyl" if comp == "aldehyde" else comp
        print(
            f"  {label:<11}: {tried:>8,} / {library:>11,}  ({_as_pct(tried, library)})",
            flush=True,
        )
    print(
        f"  {'products':<11}: {n_products:>8,} / {total_products:>11.3e}  "
        f"({_as_pct(n_products, total_products)})",
        flush=True,
    )


def _as_pct(part, whole):
    """Percentages here span ~50% down to 1e-15, so switch notation rather than
    printing a column of zeroes."""
    if whole == 0:
        return "n/a"
    pct = 100.0 * part / whole
    return f"{pct:.4f}%" if pct >= 0.0001 else f"{pct:.2e}%"


def write_summary(output_dir, run_name, unit_list, maximize):
    """
    Collapse every candidate scored during the run into one ranked table.

    Reads back the per-generation log rather than tracking in memory, so a run
    resumed from a restart file still reports everything it has ever scored.
    Scoring is deterministic, so a genome seen in several generations carries
    the same score each time and collapses to a single row.
    """
    full_filename = os.path.join(output_dir, "full_analysis_" + run_name + ".csv")

    by_genome = {}
    with open(full_filename, newline="") as full_file:
        for row in csv.DictReader(full_file):
            by_genome[row["individual"]] = (
                float(row["score"]),
                float(row["score_std"]),
            )

    ranked = sorted(by_genome.items(), key=lambda kv: kv[1][0], reverse=maximize)

    summary_filename = os.path.join(output_dir, "summary_" + run_name + ".csv")
    with open(summary_filename, mode="w+", newline="") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(
            [
                "rank",
                "score",
                "score_std",
                "individual",
                "carbonyl_smiles",
                "acid_smiles",
                "amine_smiles",
                "isocyanide_smiles",
                "product_smiles",
            ]
        )
        for rank, (genome, (score, score_std)) in enumerate(ranked, start=1):
            poly = ast.literal_eval(genome)
            writer.writerow(
                [
                    rank,
                    score,
                    score_std,
                    genome,
                    unit_list["aldehyde"].iloc[poly[0], 0],
                    unit_list["acid"].iloc[poly[1], 0],
                    unit_list["amine"].iloc[poly[2], 0],
                    unit_list["isocyanide"].iloc[poly[3], 0],
                    Chem.MolToSmiles(utils.make_molecule(poly, unit_list)),
                ]
            )

    return summary_filename, len(ranked)


def next_gen(params):
    """
    Runs the next generation of the GA

    Paramaters
    ----------
    params: list
        list with specific order
        params = [pop_size, unit_list, gen_counter, fitness_list, selection_method, mutation_rate, elitism_perc, run_name, scoring_prop]

    Returns
    --------
    params: list
        list with specific order
        params = [pop_size, unit_list, gen_counter, fitness_list, selection_method, mutation_rate, elitism_perc, run_name, scoring_prop]
    """
    pop_size = params[0]
    unit_list = params[1]
    gen_counter = params[2]
    fitness_list = params[3]
    selection_method = params[4]
    mutation_rate = params[5]
    elitism_perc = params[6]
    run_name = params[7]
    scoring_prop = params[8]
    checkpoint_dirs = params[9]
    ref_features_row = params[10]
    ref_features_cols = params[11]
    maximize = params[12]
    output_dir = params[13]
    block_freq = params[14]
    spear_counter = params[15]
    spear_thresh = params[16]
    n_jobs = params[18]

    gen_counter += 1
    ranked_population = fitness_list[1]
    ranked_scores = fitness_list[0]

    # Select percentage of top performers for next generation - "elitism"
    elitist_population = elitist_select(ranked_population, elitism_perc)

    # Selection, Crossover & Mutation
    new_population = select_crossover_mutate(
        ranked_population,
        ranked_scores,
        elitist_population,
        selection_method,
        mutation_rate,
        scoring_prop,
        pop_size,
        unit_list,
        maximize,
    )

    fitness_list = scoring.fitness_function(
        new_population,
        checkpoint_dirs,
        unit_list,
        ref_features_row,
        ref_features_cols,
        maximize,
        n_jobs,
    )

    median = int((len(fitness_list[0]) - 1) / 2)
    min_score = min(fitness_list[0])
    med_score = fitness_list[0][median]
    max_score = max(fitness_list[0])

    # Convergence: compare which building blocks the population favours now
    # against the previous generation. Once that leaderboard stops reshuffling
    # for conv_gen generations in a row, the search has settled.
    old_ranks = utils.top_ranked_blocks(block_freq)
    block_freq = utils.update_block_freq(new_population, block_freq)
    new_ranks = utils.top_ranked_blocks(block_freq)

    if len(old_ranks) < 10 or len(new_ranks) < 10:
        # too few distinct blocks seen yet for the comparison to mean anything
        spear = 0.0
    else:
        spear = spearmanr(old_ranks, new_ranks)[0]
        if np.isnan(spear):
            spear = 0.0

    if spear > spear_thresh:
        spear_counter += 1
    else:
        spear_counter = 0

    quick_filename = os.path.join(output_dir, "quick_analysis_" + run_name + ".csv")
    with open(quick_filename, mode="a+") as quick_file:
        quick_writer = csv.writer(quick_file)
        quick_writer.writerow([gen_counter, min_score, med_score, max_score, spear, spear_counter])

    for x in range(len(fitness_list[0])):
        poly = fitness_list[1][x]
        score = fitness_list[0][x]
        score_std = fitness_list[2][x]

        full_filename = os.path.join(output_dir, "full_analysis_" + run_name + ".csv")
        with open(full_filename, mode="a+") as full_file:
            full_writer = csv.writer(full_file)
            full_writer.writerow([gen_counter, poly, score, score_std])

    params = [
        pop_size,
        unit_list,
        gen_counter,
        fitness_list,
        selection_method,
        mutation_rate,
        elitism_perc,
        run_name,
        scoring_prop,
        checkpoint_dirs,
        ref_features_row,
        ref_features_cols,
        maximize,
        output_dir,
        block_freq,
        spear_counter,
        spear_thresh,
        spear,
        n_jobs,
    ]

    return params


def parent_select(ranked_population, ranked_scores, selection_method, scoring_prop, maximize):
    """
    Selects two parents. Method of selection depends on selection_method

    Parameters
    ----------
    ranked_population: list
        ordered list of polymer names, ranked from best to worst
    ranked_scores: list
        ordered list of scores, ranked from best to worst
    selection_method: str
        Type of selection operation. Options are 'random', 'random_top50', 'tournament', 'roulette', 'SUS', or 'rank'
    scoring_prop: str
        Fitness function property. Options are 'polar', 'opt_bg', 'solv_eng'

    Returns
    -------
    parents: list
        list of length 2 containing the two parent indicies
    """
    # randomly select parents
    if selection_method == "random":
        parents = []
        # randomly select two parents (as indexes from population) to cross
        parent_a = random.randint(0, len(ranked_population) - 1)
        parent_b = random.randint(0, len(ranked_population) - 1)
        # ensure parents are unique indiviudals
        if len(ranked_population) > 1:
            while parent_b == parent_a:
                parent_b = random.randint(0, len(ranked_population) - 1)

        parents.append(ranked_population[parent_a])
        parents.append(ranked_population[parent_b])

    # randomly select parents from top 50% of population
    elif selection_method == "random_top50":
        parents = []
        # randomly select two parents (as indexes from population) to cross
        parent_a = random.randint(0, len(ranked_population) / 2 - 1)
        parent_b = random.randint(0, len(ranked_population) / 2 - 1)
        # ensure parents are unique indiviudals
        if len(ranked_population) > 1:
            while parent_b == parent_a:
                parent_b = random.randint(0, len(ranked_population) / 2 - 1)

        parents.append(ranked_population[parent_a])
        parents.append(ranked_population[parent_b])

    # 3-way tournament selection
    elif selection_method == "tournament_3":
        parents = []
        # select 2 parents
        while len(parents) < 2:
            individuals = []
            # select random individual 1
            individual_1 = random.randint(0, len(ranked_population) - 1)
            individuals.append(individual_1)

            # select random individual 2
            individual_2 = random.randint(0, len(ranked_population) - 1)
            while individual_1 == individual_2:
                individual_2 = random.randint(0, len(ranked_population) - 1)
            individuals.append(individual_2)

            # select random individual 3
            individual_3 = random.randint(0, len(ranked_population) - 1)
            while (individual_3 == individual_1) or (individual_3 == individual_2):
                individual_3 = random.randint(0, len(ranked_population) - 1)
            individuals.append(individual_3)
            # Make list of their fitness
            scores = [
                ranked_scores[individual_1],
                ranked_scores[individual_2],
                ranked_scores[individual_3],
            ]

            # find the index of the best fitness score
            best_index = np.argmax(scores) if maximize else np.argmin(scores)

            best_individual = individuals[best_index]
            parent = ranked_population[best_individual]

            if parent not in parents:
                parents.append(parent)

    # 4-way tournament selection
    elif selection_method == "tournament_4":
        parents = []
        # select 2 parents
        while len(parents) < 2:
            individuals = []
            # select random individual 1
            individual_1 = random.randint(0, len(ranked_population) - 1)
            individuals.append(individual_1)

            # select random individual 2
            individual_2 = random.randint(0, len(ranked_population) - 1)
            while individual_1 == individual_2:
                individual_2 = random.randint(0, len(ranked_population) - 1)
            individuals.append(individual_2)

            # select random individual 3
            individual_3 = random.randint(0, len(ranked_population) - 1)
            while (individual_3 == individual_1) or (individual_3 == individual_2):
                individual_3 = random.randint(0, len(ranked_population) - 1)
            individuals.append(individual_3)

            # select random individual 4
            individual_4 = random.randint(0, len(ranked_population) - 1)
            while (
                (individual_4 == individual_1)
                or (individual_4 == individual_2)
                or (individual_4 == individual_3)
            ):
                individual_4 = random.randint(0, len(ranked_population) - 1)
            individuals.append(individual_4)
            # Make list of their fitness
            scores = [
                ranked_scores[individual_1],
                ranked_scores[individual_2],
                ranked_scores[individual_3],
                ranked_scores[individual_4],
            ]

            # find the index of the best fitness score
            best_index = np.argmax(scores) if maximize else np.argmin(scores)

            best_individual = individuals[best_index]
            parent = ranked_population[best_individual]

            if parent not in parents:
                parents.append(parent)

    elif selection_method == "tournament_2":
        parents = []
        # select 2 parents
        while len(parents) < 2:
            individuals = []
            # select random individual 1
            individual_1 = random.randint(0, len(ranked_population) - 1)
            individuals.append(individual_1)

            # select random individual 2
            individual_2 = random.randint(0, len(ranked_population) - 1)
            while individual_1 == individual_2:
                individual_2 = random.randint(0, len(ranked_population) - 1)
            individuals.append(individual_2)

            # Make list of their fitness
            scores = [ranked_scores[individual_1], ranked_scores[individual_2]]

            # find the index of the best fitness score
            best_index = np.argmax(scores) if maximize else np.argmin(scores)

            best_individual = individuals[best_index]
            parent = ranked_population[best_individual]

            if parent not in parents:
                parents.append(parent)

    elif (
        selection_method == "roulette"
    ):  # ranked_population, ranked_scores, selection_method, scoring_prop
        parents = []
        # create wheel
        wheel = []
        # bottom limit
        limit = 0

        if maximize:
            # higher score = bigger slice
            total = sum(ranked_scores)
            for x in range(len(ranked_scores)):
                # fitness proportion
                fitness = ranked_scores[x] / total
                # appends the bottom and top limits of the pie, the score, and the polymer
                wheel.append((limit, limit + fitness, ranked_scores[x], ranked_population[x]))
                limit += fitness

        else:
            # lower score = bigger slice
            inversed_scores = [1 / x for x in ranked_scores]
            total = sum(inversed_scores)
            for x in range(len(inversed_scores)):
                # fitness proportion
                fitness = inversed_scores[x] / total
                # appends the bottom and top limits of the pie, the score, and the polymer
                wheel.append((limit, limit + fitness, inversed_scores[x], ranked_population[x]))
                limit += fitness

        # random number between 0 and 1
        r = random.random()
        # search for polymer in that pie
        score, polymer = utils.binSearch(wheel, r)
        parents.append(polymer)
        while len(parents) < 2:
            r = random.random()
            score, polymer = utils.binSearch(wheel, r)
            if polymer not in parents:
                parents.append(polymer)

    elif selection_method == "SUS":  # stochastic universal sampling
        parents = []
        # create wheel
        wheel = []
        # bottom limit
        limit = 0

        if maximize:
            # higher score = bigger slice
            total = sum(ranked_scores)
            for x in range(len(ranked_scores)):
                # fitness proportion
                fitness = ranked_scores[x] / total
                # appends the bottom and top limits of the pie, the score, and the polymer
                wheel.append((limit, limit + fitness, ranked_scores[x], ranked_population[x]))
                limit += fitness

        else:
            # lower score = bigger slice
            inversed_scores = [1 / x for x in ranked_scores]
            total = sum(inversed_scores)
            for x in range(len(inversed_scores)):
                # fitness proportion
                fitness = inversed_scores[x] / total
                # appends the bottom and top limits of the pie, the score, and the polymer
                wheel.append((limit, limit + fitness, inversed_scores[x], ranked_population[x]))
                limit += fitness

        # separation between selected points on wheel
        stepSize = 0.5
        # random number between 0 and 1
        r = random.random()

        # search for polymer in that pie
        score, polymer = utils.binSearch(wheel, r)
        parents.append(polymer)
        while len(parents) < 2:
            r += stepSize
            if r > 1:
                r %= 1
            score, polymer = utils.binSearch(wheel, r)
            parents.append(polymer)

    elif selection_method == "rank":
        parents = []
        # reverse population so worst scores get smallest slice of pie
        ranked_population.reverse()

        # creates wheel
        wheel = []
        # total sum of the indicies
        total = sum(range(1, len(ranked_population) + 1))
        top = 0
        for p in range(len(ranked_population)):
            # adds 1 so the first element does not have a pie slice of size 0
            x = p + 1
            # fraction of total pie
            f = x / total
            wheel.append((top, top + f, ranked_population[p]))
            top += f

        # pick random parents from wheel
        while len(parents) < 2:
            r = random.random()
            polymer = utils.rank_binSearch(wheel, r)

            if polymer not in parents:
                parents.append(polymer)
    else:
        print("not a valid selection method")

    return parents


def select_crossover_mutate(
    ranked_population,
    ranked_scores,
    elitist_population,
    selection_method,
    mutation_rate,
    scoring_prop,
    pop_size,
    unit_list,
    maximize,
):
    """
    Perform selection, crossover, and mutation operations

    Parameters
    ----------
    ranked_population: list
        ordered list containing lists of polymers of format [mon_1_index, mon_2_index]
    ranked_scores: list
        ordered list of scores of population
    elitist_population: list
        list of elite polymers to pass to next generation
    selection_method: str
        Type of selection operation. Options are 'random', 'tournament', 'roulette', 'SUS', or 'rank'
    mutation_rate: int
        Chance of polymer to undergo mutation
    scoring_prop: str
        Fitness function property. Options are 'polar', 'opt_bg', 'solv_eng'
    pop_size: int
        number of individuals in the population
    unit_list: Dataframe
        dataframe containing the SMILES of all monomers

    Returns
    -------
    new_pop: list
        list of new polymers of format [mon_1_index, mon_2_index]
    new_pop_smiles: list
        list of the SMILES of the new polymers
    """

    new_pop = deepcopy(elitist_population)

    # loop until enough children have been added to reach population size
    while len(new_pop) < pop_size:
        # select two parents
        parents = parent_select(
            ranked_population, ranked_scores, selection_method, scoring_prop, maximize
        )

        # create hybrid child
        temp_child = []

        # take first unit from parent 1 and second unit from parent 2
        for i in range(4):
            source = random.randint(0, 1)
            temp_child.append(parents[source][i])

        # give child opportunity for mutation
        temp_child = mutate(temp_child, unit_list, mutation_rate)

        # check for duplication
        if temp_child in new_pop:
            pass
        elif utils.not_valid(temp_child, unit_list):
            pass
        else:
            new_pop.append(temp_child)

    return new_pop


def mutate(temp_child, unit_list, mut_rate):
    rand = random.randint(1, 100)
    if rand > (mut_rate * 100):
        return temp_child

    point = random.randint(0, 3)  # 0=ald, 1=acid, 2=amine, 3=iso
    components = COMPONENT_SLOTS
    comp = components[point]
    temp_child[point] = random.randint(0, len(unit_list[comp]) - 1)
    return temp_child


def elitist_select(ranked_population, elitism_perc):
    """
    Selects a percentage of the top polymers to ensure in next generation

    Parameters
    ----------
    ranked_population: list
        ordered list containing lists of polymers of format [mon_1_index, mon_2_index]
    elitism_perc: float
        percentage of generation to pass to next generation. Can range from 0-1 (although 1 is the entire generation)

    Returns
    -------
    elitist_list: list
        list of polymers each of format [mon_1_index, mon_2_index]
    """

    # find number of parents to pass to next generation
    elitist_count = int(len(ranked_population) * elitism_perc)
    elitist_list = []

    for x in range(elitist_count):
        elitist_list.append(ranked_population[x])

    return elitist_list


def init_gen(
    pop_size,
    selection_method,
    mutation_rate,
    elitism_perc,
    run_name,
    scoring_prop,
    unit_list,
    checkpoint_dirs,
    ref_features_row,
    ref_features_cols,
    maximize,
    output_dir,
    spear_thresh,
    n_jobs,
):
    """
    Create initial population

    Parameters
    -----------
    pop_size: int
        number of individuals in the population
    selection_method: str
        Type of selection operation. Options are 'random', 'tournament', 'roulette', 'SUS', or 'rank'
    mutation_rate: int
        Chance of polymer to undergo mutation
    elitism_perc: float
        percentage of generation to pass to next generation. Can range from 0-1 (although 1 is the entire generation)
    run_name: str
        name of this GA run
    scoring_prop: str
        Fitness function property. Options are 'polar', 'opt_bg', 'solv_eng'
    unit_list: dict
        dict of dataframes keyed by component type (aldehyde, acid, amine, isocyanide)

    Returns
    -------
    params: list
        format is [pop_size, unit_list, gen_counter, fitness_list, selection_method, mutation_rate, elitism_perc, run_name, scoring_prop]
    """
    # initialize generation counter
    gen_counter = 1

    # create initial population as list of polymers
    population = []

    while len(population) < pop_size:
        temp_poly = []
        components = COMPONENT_SLOTS
        for comp in components:
            idx = random.randint(0, len(unit_list[comp]) - 1)
            temp_poly.append(idx)

        if temp_poly in population:
            continue
        elif utils.not_valid(temp_poly, unit_list):
            pass
        else:
            population.append(temp_poly)

    # create new analysis files
    quick_filename = os.path.join(output_dir, "quick_analysis_" + run_name + ".csv")
    with open(quick_filename, mode="w+") as quick:
        quick_writer = csv.writer(quick)
        quick_writer.writerow(
            ["gen", "min_score", "med_score", "max_score", "spearman", "spear_counter"]
        )

    full_filename = os.path.join(output_dir, "full_analysis_" + run_name + ".csv")
    with open(full_filename, mode="w+") as full:
        full_writer = csv.writer(full)
        full_writer.writerow(["gen", "individual", "score", "score_std"])

    fitness_list = scoring.fitness_function(
        population,
        checkpoint_dirs,
        unit_list,
        ref_features_row,
        ref_features_cols,
        maximize,
        n_jobs,
    )

    median = int((len(fitness_list[0]) - 1) / 2)
    min_score = min(fitness_list[0])
    med_score = fitness_list[0][median]
    max_score = max(fitness_list[0])

    quick_filename = os.path.join(output_dir, "quick_analysis_" + run_name + ".csv")
    with open(quick_filename, mode="a+") as quick_file:
        quick_writer = csv.writer(quick_file)
        quick_writer.writerow([1, min_score, med_score, max_score, 0.0, 0])

    for x in range(len(fitness_list[0])):
        poly = fitness_list[1][x]
        score = fitness_list[0][x]
        score_std = fitness_list[2][x]

        full_filename = os.path.join(output_dir, "full_analysis_" + run_name + ".csv")
        with open(full_filename, mode="a+") as full_file:
            full_writer = csv.writer(full_file)
            full_writer.writerow([gen_counter, poly, score, score_std])

    params = [
        pop_size,
        unit_list,
        gen_counter,
        fitness_list,
        selection_method,
        mutation_rate,
        elitism_perc,
        run_name,
        scoring_prop,
        checkpoint_dirs,
        ref_features_row,
        ref_features_cols,
        maximize,
        output_dir,
        utils.update_block_freq(population, {}),
        0,
        spear_thresh,
        0.0,
        n_jobs,
    ]

    return params


if __name__ == "__main__":
    usage = "usage: %prog [options] "
    parser = argparse.ArgumentParser(usage)

    # sets input arguments
    # free-text label for this run, used in output filenames
    parser.add_argument("chem_property", action="store", type=str)
    # 'A', 'B', 'C', 'D', or 'E'
    parser.add_argument("run_label", action="store", type=str)
    # path to all_data_extra_x.csv used to compute the reference formulation features
    parser.add_argument("features_ref_path", action="store", type=str)
    # one or more chemprop v1 checkpoint directories (trained_model_checkpoints/), one per CV fold
    parser.add_argument("checkpoint_dirs", nargs="+", type=str)
    # optional: number of generations (default 500; use a small value for test runs)
    parser.add_argument("--n_generations", type=int, default=500)
    # component building-block CSV files (must each have a 'smiles' column)
    parser.add_argument("--aldehydes", required=True, type=str)
    parser.add_argument("--acids", required=True, type=str)
    parser.add_argument("--amines", required=True, type=str)
    parser.add_argument("--isocyanides", required=True, type=str)
    # optimize for higher predicted values instead of lower
    parser.add_argument("--maximize", action="store_true")
    # directory to write every artifact of this run into (created if absent)
    parser.add_argument("--output_dir", required=True, type=str)

    # GA hyperparameters. Defaults are the best-practice values from
    # J. Chem. Phys. 159, 091501 (2023), and can now be set independently.
    parser.add_argument("--pop_size", type=int, default=32)
    parser.add_argument(
        "--selection_method",
        type=str,
        default="tournament_3",
        choices=[
            "random",
            "random_top50",
            "tournament_2",
            "tournament_3",
            "tournament_4",
            "roulette",
            "rank",
            "SUS",
        ],
    )
    parser.add_argument("--mutation_rate", type=float, default=0.4)
    parser.add_argument("--elitism_perc", type=float, default=0.5)
    # self-termination: stop once the favoured building blocks stop reshuffling
    parser.add_argument("--spear_thresh", type=float, default=0.8)
    parser.add_argument("--conv_gen", type=int, default=50)
    # how many ensemble members to score concurrently. Most of a member's wall
    # time is interpreter start-up, so this is close to a linear speed-up until
    # it reaches the number of models. Set it to your allocated core count.
    parser.add_argument("--n_jobs", type=int, default=1)

    args = parser.parse_args()

    main(
        args.chem_property,
        args.run_label,
        args.checkpoint_dirs,
        args.features_ref_path,
        args.n_generations,
        args.aldehydes,
        args.acids,
        args.amines,
        args.isocyanides,
        args.maximize,
        args.output_dir,
        args.pop_size,
        args.selection_method,
        args.mutation_rate,
        args.elitism_perc,
        args.spear_thresh,
        args.conv_gen,
        args.n_jobs,
    )
