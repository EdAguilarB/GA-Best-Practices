# Imports
import csv
import random
import numpy as np
from copy import deepcopy
import pickle
import argparse
import torch

import utils
import scoring


def main(parameter_type, parameter, chem_property, run_label, model_path):
    # reuse initial state
    initial_restart = "y"
    # flag for restart from save
    restart = "n"
    # Run number (for use with same initial states), can be A, B, C, D, or E
    # run_label = 'D'

    # scoring property. Can be 'polar', 'opt_bg', or 'solv_eng'
    # scoring_prop = 'polar'

    if parameter_type == "pop_size":
        # number of polymers in population, can be 16, 32, 48, 64, 80, 96
        pop_size = int(parameter)

        selection_method = "random"
        mutation_rate = 0.4
        elitism_perc = 0.5

    elif parameter_type == "selection_method":
        # selection method. Can be 'random', 'tournament_2', 'tournament_3', 'tournament_4', 'roulette', 'rank', 'SUS'
        selection_method = parameter

        pop_size = 32
        mutation_rate = 0.4
        elitism_perc = 0.5

    elif parameter_type == "mutation_rate":
        # mutation rate. Can be 0.1-0.9, in increments of 0.1
        mutation_rate = float(parameter)

        pop_size = 32
        selection_method = "random"
        elitism_perc = 0.5

        parameter = int(mutation_rate * 100)

    elif parameter_type == "elitism_perc":
        # elitism percentage. Percentage of top candidates to pass on to next generation. Can be 0, 0.25, 0.5
        elitism_perc = float(parameter)

        pop_size = 32
        selection_method = "random"
        mutation_rate = 0.4

        parameter = int(elitism_perc * 100)

    else:
        print("not a valid parameter type")

    # Number of generations to run
    n_generations = 500

    # GA run file name, with the format of "parameter_changed parameter_value fitness_property run_label(ABCDE)"
    run_name = (
        parameter_type + "_" + str(parameter) + "_" + chem_property + "_" + run_label
    )

    # Load pre-trained scoring model
    model = torch.load(model_path, weights_only=False)
    model.eval()

    # Create list of possible building block unit SMILES in specific format
    unit_list = utils.make_unit_list()

    if restart == "y":
        # reload parameters and random state from restart file
        last_gen_filename = "../last_gen_params/last_gen_" + run_name + ".p"
        open_params = open(last_gen_filename, "rb")
        params = pickle.load(open_params)
        open_params.close()

        # inject model (not stored in pickle; always loaded fresh from model_path)
        if len(params) <= 9:
            params.append(model)
        else:
            params[9] = model

        randstate_filename = "../rand_states/randstate_" + run_name + ".p"
        open_rand = open(randstate_filename, "rb")
        randstate = pickle.load(open_rand)
        random.setstate(randstate)
        open_rand.close()

    else:
        if initial_restart == "n":
            # sets initial state
            randstate = random.getstate()
            initial_randstate = (
                "../initial_randstates/initial_randstate_" + run_label + ".p"
            )
            rand_file = open(initial_randstate, "wb")
            pickle.dump(randstate, rand_file)
            rand_file.close()
        else:
            # re-opens intial state during troubleshooting
            initial_randstate = (
                "/ihome/ghutchison/blp62/GA_best_practices/initial_randstates/initial_randstate_"
                + run_label
                + ".p"
            )
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
            model,
        )

        # pickle parameters needed for restart
        last_gen_filename = "../last_gen_params/last_gen_" + run_name + ".p"
        params_file = open(last_gen_filename, "wb")
        pickle.dump(params, params_file)
        params_file.close()

        # pickle random state for restart
        randstate = random.getstate()
        randstate_filename = "../rand_states/randstate_" + run_name + ".p"
        rand_file = open(randstate_filename, "wb")
        pickle.dump(randstate, rand_file)
        rand_file.close()

    for _ in range(n_generations):
        # run next generation of GA
        params = next_gen(params)

        # pickle parameters needed for restart
        last_gen_filename = "../last_gen_params/last_gen_" + run_name + ".p"
        params_file = open(last_gen_filename, "wb")
        pickle.dump(params, params_file)
        params_file.close()

        # pickle random state for restart
        randstate = random.getstate()
        randstate_filename = "../rand_states/randstate" + run_name + ".p"
        rand_file = open(randstate_filename, "wb")
        pickle.dump(randstate, rand_file)
        rand_file.close()


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
    model = params[9]

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
    )

    fitness_list = scoring.fitness_function(new_population, model, unit_list)

    min_score = fitness_list[0][0]
    median = int((len(fitness_list[0]) - 1) / 2)
    med_score = fitness_list[0][median]
    max_score = fitness_list[0][-1]

    quick_filename = "../quick_files/quick_analysis_" + run_name + ".csv"
    with open(quick_filename, mode="a+") as quick_file:
        quick_writer = csv.writer(quick_file)
        quick_writer.writerow([gen_counter, min_score, med_score, max_score])

    for x in range(len(fitness_list[0])):
        poly = fitness_list[1][x]
        score = fitness_list[0][x]

        full_filename = "../full_files/full_analysis_" + run_name + ".csv"
        with open(full_filename, mode="a+") as full_file:
            full_writer = csv.writer(full_file)
            full_writer.writerow([gen_counter, poly, score])

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
        model,
    ]

    return params


def parent_select(ranked_population, ranked_scores, selection_method, scoring_prop):
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
            if scoring_prop == "polar":
                best_index = np.argmax(scores)
            else:
                best_index = np.argmin(scores)

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
            if scoring_prop == "polar":
                best_index = np.argmax(scores)
            else:
                best_index = np.argmin(scores)

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
            if scoring_prop == "polar":
                best_index = np.argmax(scores)
            else:
                best_index = np.argmin(scores)

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

        if scoring_prop == "opt_bg":
            inversed_scores = [1 / x for x in ranked_scores]
            total = sum(inversed_scores)

            for x in range(len(inversed_scores)):
                # fitness proportion
                fitness = inversed_scores[x] / total
                # appends the bottom and top limits of the pie, the score, and the polymer
                wheel.append(
                    (limit, limit + fitness, inversed_scores[x], ranked_population[x])
                )
                limit += fitness

        elif scoring_prop == "polar":
            # sum of scores
            total = sum(ranked_scores)
            for x in range(len(ranked_scores)):
                # fitness proportion
                fitness = ranked_scores[x] / total
                # appends the bottom and top limits of the pie, the score, and the polymer
                wheel.append(
                    (limit, limit + fitness, ranked_scores[x], ranked_population[x])
                )
                limit += fitness

        else:
            print(
                "selection method does not work with negative numbers (solvation ratios)"
            )

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

        if scoring_prop == "opt_bg":
            inversed_scores = [1 / x for x in ranked_scores]
            total = sum(inversed_scores)

            for x in range(len(inversed_scores)):
                # fitness proportion
                fitness = inversed_scores[x] / total
                # appends the bottom and top limits of the pie, the score, and the polymer
                wheel.append(
                    (limit, limit + fitness, inversed_scores[x], ranked_population[x])
                )
                limit += fitness

        elif scoring_prop == "polar":
            # sum of scores
            total = sum(ranked_scores)
            for x in range(len(ranked_scores)):
                # fitness proportion
                fitness = ranked_scores[x] / total
                # appends the bottom and top limits of the pie, the score, and the polymer
                wheel.append(
                    (limit, limit + fitness, ranked_scores[x], ranked_population[x])
                )
                limit += fitness

        else:
            print(
                "selection method does not work with negative numbers (solvation ratios)"
            )

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
            ranked_population, ranked_scores, selection_method, scoring_prop
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
        elif utils.not_valid(temp_child, unit_list) == True:
            pass
        else:
            new_pop.append(temp_child)

    return new_pop


def mutate(temp_child, unit_list, mut_rate):
    rand = random.randint(1, 100)
    if rand > (mut_rate * 100):
        return temp_child

    point = random.randint(0, 3)  # 0=ald, 1=acid, 2=amine, 3=iso
    components = ["aldehyde", "acid", "amine", "isocyanide"]
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
    model,
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
        components = ["aldehyde", "acid", "amine", "isocyanide"]
        for comp in components:
            idx = random.randint(0, len(unit_list[comp]) - 1)
            temp_poly.append(idx)

        if temp_poly in population:
            continue
        elif utils.not_valid(temp_poly, unit_list) == True:
            pass
        else:
            population.append(temp_poly)

    # create new analysis files
    quick_filename = "../quick_files/quick_analysis_" + run_name + ".csv"
    with open(quick_filename, mode="w+") as quick:
        quick_writer = csv.writer(quick)
        quick_writer.writerow(["gen", "min_score", "med_score", "max_score"])

    full_filename = "../full_files/full_analysis_" + run_name + ".csv"
    with open(full_filename, mode="w+") as full:
        full_writer = csv.writer(full)
        full_writer.writerow(["gen", "individual", "score"])

    fitness_list = scoring.fitness_function(population, model, unit_list)

    min_score = fitness_list[0][0]
    median = int((len(fitness_list[0]) - 1) / 2)
    med_score = fitness_list[0][median]
    max_score = fitness_list[0][-1]

    quick_filename = "../quick_files/quick_analysis_" + run_name + ".csv"
    with open(quick_filename, mode="a+") as quick_file:
        quick_writer = csv.writer(quick_file)
        quick_writer.writerow([1, min_score, med_score, max_score])

    for x in range(len(fitness_list[0])):
        poly = fitness_list[1][x]
        score = fitness_list[0][x]

        full_filename = "../full_files/full_analysis_" + run_name + ".csv"
        with open(full_filename, mode="a+") as full_file:
            full_writer = csv.writer(full_file)
            full_writer.writerow([gen_counter, poly, score])

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
        model,
    ]

    return params


if __name__ == "__main__":
    usage = "usage: %prog [options] "
    parser = argparse.ArgumentParser(usage)

    # sets input arguments
    # parameter type = 'pop_size', 'selection_method', 'mutation_rate', or 'elitism_perc'
    parser.add_argument("parameter_type", action="store", type=str)
    # the value to change the paraemter type to
    parser.add_argument("parameter", action="store", type=str)
    # 'polar', 'opt_bg', or 'solv_eng' — controls optimization direction in selection methods
    parser.add_argument("chem_property", action="store", type=str)
    # 'A', 'B', 'C', 'D', or 'E'
    parser.add_argument("run_label", action="store", type=str)
    # path to a saved PyTorch model file
    parser.add_argument("model_path", action="store", type=str)

    args = parser.parse_args()

    main(args.parameter_type, args.parameter, args.chem_property, args.run_label, args.model_path)
