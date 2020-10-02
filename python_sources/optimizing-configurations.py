# The script has 2 parts:
#
# Part 1 - generate random solutions, by repeatedly generating a bag configuration
#          and filling bags until there are not enough toys.  
#
# Part 2- optimize the best solutions from part 1 by mutating a random configuration
#         in a random solution via adding/removing/swapping a toy and keeping mutations
#         that improve the score.  

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import sys
import re
from datetime import datetime
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

NUM_SAMPLES = 4000
NUM_SOLUTIONS = 5000
NUM_SOLNS_TO_OPT = 25
NUM_OPT_ITERS=40000

def main(num_samples=NUM_SAMPLES, num_solutions=NUM_SOLUTIONS, num_solns_to_opt=NUM_SOLNS_TO_OPT, num_opt_iters=NUM_OPT_ITERS): 
    print("*********** Step 1 - generating {} random solutions **************".format(num_solutions))
    all_solns = generate_solutions(num_solutions)
    best_solns = sorted(all_solns, key=lambda x:-x["total_score"])[:num_solns_to_opt]  
    
    print("*********** Step 2 - optimizing top {} solutions **************".format(num_solns_to_opt))
    optimized_best_solns = optimize_solutions(best_solns, num_opt_iters)
    best_soln = optimized_best_solns[0]
    
    # Write best solution as submission
    write_solution(best_soln)

def load_gifts_and_toys():
    toys = {
        "horse":  { "sample": lambda: max(0, np.random.normal(5,2,1)[0]), "sample_type": "normal(5,2)" },
        "ball":   { "sample": lambda: max(0, 1 + np.random.normal(1,0.3,1)[0]), "sample_type": "normal(1,0.3)" },
        "bike":   { "sample": lambda: max(0, np.random.normal(20,10,1)[0]), "sample_type": "normal(20,10)" },
        "train":  { "sample": lambda: max(0, np.random.normal(10,5,1)[0]), "sample_type": "normal(10,5)" },
        "coal":   { "sample": lambda: 47 * np.random.beta(0.5,0.5,1)[0], "sample_type": "47*beta(0.5,0.5)" },
        "book":   { "sample": lambda: np.random.chisquare(2,1)[0], "sample_type": "chi(2)" },
        "doll":   { "sample": lambda: np.random.gamma(5,1,1)[0], "sample_type": "gamma(5,1)" },
        "block":  { "sample": lambda: np.random.triangular(5,10,20,1)[0], "sample_type": "triagl(5,10,20)" },
        "gloves": { "sample": lambda: 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0], "sample_type": "0.3:3+rand(1), 0.7:rand(1)" },
    }
    
    gifts_df = pd.read_csv("../input/gifts.csv", sep=",")
    gifts = gifts_df["GiftId"].values
    
    for t in toys:
        ids = [g for g in gifts if t in g.split("_")[0]]
        toys[t]["ids"] = ids
        toys[t]["count"] = len(ids)
    
    return gifts, toys
    
def shuf(ar):
    random.shuffle(ar)
    return ar

# generate weight samples for bags up to size MAX_DEPTH in the cache tree
def generate_cache_tree(toys, num_samples, max_depth):
    cache_tree = {}
    queue = [[t] for t in list(toys)]
    curr_depth = 0
    for curr_iter in range(10000000):
        if len(queue) == 0:
            break

        bag = queue.pop(0)

        if len(bag) > curr_depth:
            curr_depth = len(bag)
            #print("moving to depth {} in BFS".format(curr_depth))

        #if curr_iter > 0 and curr_iter % 50 == 0:
        #    print("iter: {:8}, config:{}".format(curr_iter, bag))

        # generate samples
        samples = [sum([toys[x]["sample"]() for x in bag]) for _ in range(num_samples)]

        # generate new nodes if we haven't reached max depth
        if len(bag) < max_depth:
            new_bags = [bag + [t] for t in list(toys) if t >= bag[-1]] # prevent a,b and b,a comparisons using lex sorting
            for new_bag in new_bags:
                queue.append(new_bag)

        # get leaf from cache tree
        node = cache_tree
        for t in bag[:-1]:
            if t not in node:
                node[t] = {"children": {}}
            node = node[t]["children"]

        node[bag[-1]] = {"value": {"samples": np.array(samples)}, "children": {}}
        
    return cache_tree
    
def getSamplesForBag(bag):
    all_samples = []
    node = None
    match = []
    
    for t in sorted(bag):
        if node is None:
            node = cache_tree[t]
            match.append(t)
        elif t in node["children"]:
            node = node["children"][t]
            match.append(t)
        else:
            all_samples.append((match[:], node["value"]["samples"]))
            node, match = cache_tree[t], [t]
        
    # if node is not None, then we had a partial match we need to save
    if node is not None:
        all_samples.append((match, node["value"]["samples"]))
        node, match = None, []
        
    return np.sum([x[1][random_indexes[random.randint(0,len(random_indexes)-1)]] for x in all_samples], axis=0)

cache = {}
def getWeights(bags, num_simulations, use_cache=False, verbose=False):
    bags_weights = np.zeros((1000, num_simulations))
    for i, b in enumerate(bags):
        k = " ".join(b)
        if len(b) == 0:
            continue
        elif use_cache and (k in cache) and (max(cache[k]) >= num_simulations):
            if verbose: print("using cache for bag {}".format(b))
            bags_weights[i,:] = cache[k][max(cache[k])]
        else:
            weights = getSamplesForBag(b)
            bags_weights[i,:] = weights[:num_simulations]
            
            if k not in cache:
                cache[k] = {}
            cache[k][num_simulations] = weights
            
    return bags_weights

max_toy_count_per_sample = 4
max_toy_samples = {
    "horse":  max_toy_count_per_sample, 
    "ball":   max_toy_count_per_sample, 
    "bike":   min(2, max_toy_count_per_sample), 
    "train":  min(3, max_toy_count_per_sample), 
    "coal":   min(1, max_toy_count_per_sample), 
    "book":   max_toy_count_per_sample, 
    "doll":   max_toy_count_per_sample, 
    "block":  min(4, max_toy_count_per_sample), 
    "gloves": max_toy_count_per_sample
}
    
def get_toys_sample(toy_counts, sample_size):
    toys_sample = [t_i for t in toy_counts for t_i in [t]*min(max_toy_samples[t], toy_counts[t])]
    return random.sample(toys_sample, min(len(toys_sample), sample_size))

def get_new_toy_counts(toy_counts, config, num_bags_left=1000):
    config_cnts = {c:len([1 for t in config if t == c]) for c in set(config)}
    config_cnts_available_by_type = {t: math.floor(toy_counts[t] / config_cnts[t]) if toy_counts[t] > config_cnts[t] else 0 for t in config_cnts}
    total_configs_available = min(min(config_cnts_available_by_type.values()), num_bags_left)
    new_toys_counts = {t:toy_counts[t]-(config_cnts[t]*total_configs_available) if t in config_cnts else toy_counts[t] for t in toy_counts}
    return new_toys_counts, total_configs_available

def get_configs(toy_counts, num_bags_left=1000, max_toys_to_consider=9, num_simulations=10000, use_score_total=False):
    items_list = get_toys_sample(toy_counts, max_toys_to_consider)
    
    res = []
    prev_expected_score_for_bag = 0.0
    prev_expected_score = 0.0
    
    for num_toys in range(3,max_toys_to_consider):
        items = items_list[:num_toys]
        
        # run simulation
        weights = getSamplesForBag(items)[:num_simulations]
        scores_mask = weights < 50.0
        expected_score_for_bag = np.mean(scores_mask * weights)
        accept_ratio = np.mean(scores_mask)
        
        new_toy_counts, num_bags_used = get_new_toy_counts(toy_counts, items, num_bags_left=num_bags_left)
        expected_score = expected_score_for_bag * num_bags_used
        
        res.append({"config": items, "score/bag": expected_score_for_bag, "score": expected_score, "toy_counts": new_toy_counts, 
                    "num_bags": num_bags_used, "accept_ratio": accept_ratio, "num_bags_left": num_bags_left-num_bags_used})
        
        if use_score_total:
            if expected_score < prev_expected_score:
                break
            else:
                prev_expected_score = expected_score
        
        if expected_score_for_bag < prev_expected_score_for_bag:
            break
        else:    
            prev_expected_score_for_bag = expected_score_for_bag
           
    return res

def get_config(toy_counts, num_bags_left=1000, max_toys_to_consider=9, num_simulations=10000):
    return random.sample(get_configs(toy_counts, num_bags_left=num_bags_left, max_toys_to_consider=max_toys_to_consider, num_simulations=num_simulations)[-3:], 1)[0]

def add_config_to_toy_counts(config, num_bags, toy_counts):
    config_cnts = {c:len([1 for t in config if t == c]) for c in set(config)}
    orig_toy_counts = {t: toy_counts[t] + (config_cnts[t]*num_bags) if t in config_cnts else toy_counts[t] for t in toy_counts}
    return orig_toy_counts
 
def mutate_config(config_dict, swap_prob=0.8, add_prob=0.1, remove_prob=0.1, max_bag_size=10):
    config = config_dict["config"][:]
    prev_toy_counts = {k:config_dict["toy_counts"][k] for k in config_dict["toy_counts"]}
    num_bags = config_dict["num_bags"]
    toy_counts = add_config_to_toy_counts(config, num_bags, prev_toy_counts)
    num_bags_left = config_dict["num_bags_left"] + num_bags
    
    # mutate config
    action = ""
    for i in range(100):
        r = random.random()
        case = ""
        if r < swap_prob:
            idx = random.randint(0,len(config)-1)
            removed = config.pop(idx)
            added = random.sample([t for t in toy_counts if toy_counts[t] > 0], 1)[0]
            config.insert(idx, added)
            action = "swap:{} -> {}".format(removed, added)
        elif r - swap_prob < remove_prob:
            if len(config) <= 3: continue
            removed = config.pop(random.randint(0,len(config)-1))
            action = "removed:{}".format(removed)
        else:
            if len(config) >= max_bag_size: continue
            added = random.sample([t for t in toy_counts if toy_counts[t] > 0], 1)[0]
            config.append(added)
            action = "added:{}".format(added)
            
        break
    
    #calculate new stats
    new_weights = getSamplesForBag(config)
    scores_mask = new_weights < 50.0
    new_expected_score_for_bag = np.mean(scores_mask * new_weights)
    new_accept_ratio = np.mean(scores_mask)

    new_toy_counts, num_bags_used = get_new_toy_counts(toy_counts, config, num_bags_left=num_bags_left)
    new_expected_score = new_expected_score_for_bag * num_bags_used

    return {"config": config, "score/bag":new_expected_score_for_bag, "score": new_expected_score, "toy_counts": new_toy_counts, 
                 "num_bags": num_bags_used, "accept_ratio": new_accept_ratio, "num_bags_left": num_bags_left-num_bags_used, "action": action}

def mutate_path(path, num_mutations=1):
    idx = random.randint(0,len(path)-1)
    new_step = mutate_config(path[idx])
    for _ in range(1,num_mutations):
        new_step = mutate_config(new_step)
    partial_path = path[:idx] + [new_step]
    new_path = get_a_complete_path(partial_path)
    return new_path

def get_a_complete_path(path=[], i=0):
    num_bags_left = 1000 
    total_score = 0.0 
    
    for step in path:
        num_bags_used_for_step = min(num_bags_left, step["num_bags"])
        num_bags_left -= num_bags_used_for_step
        total_score += num_bags_used_for_step * step["score/bag"]
    
    if num_bags_left <= 0 or i > 100:
        if num_bags_left < 0:
            raise Exception("num_bags_left < 0, an error occurred!")
        return {"path": path, "total_score": total_score}
    
    if len(path) == 0:
        toy_counts = {t:toys[t]["count"] for t in toys}
    else:
        toy_counts = path[-1]["toy_counts"]
    
    config = get_config(toy_counts, num_bags_left)
    return get_a_complete_path(path + [config], i+1) 

def generate_solutions(num_solutions=5000):  
    all_solns = []
    best_score = 0
    for i in range(num_solutions):
        try:
            soln = get_a_complete_path([])
            score = soln["total_score"]
            all_solns.append(soln)
            if score > best_score:
                print("{:7} - new best found: score: {}".format(i, score))
                best_score = score
            if i % 1000 == 0:
                print("iter {} - best found: {}".format(i, best_score))
        except Exception as e:
            raise e
    return all_solns

def optimize_solutions(population, num_iters):

    best_score = population[0]["total_score"]
    best_solns = [population[0]]
    
    improvement_deltas = []
    
    for i in range(num_iters):
        if i > 0 and i % 1000 == 0:
            k = 1000
            last_k = improvement_deltas[-k:]
            avg_improv = np.mean(last_k)
            num_improved = len([x for x in last_k if x > 0])
            avg_per_impr = np.mean([x for x in last_k if x > 0])
            print("{} - (last {}) avg: {:.2f}, avg/pos:{:.2f}/{}".format(i, k, avg_improv, avg_per_impr, num_improved))
        
        try:
        
            idx = random.randint(0, len(population)-1)
            soln = population.pop(idx)
            score = soln["total_score"]
            
            num_mutations = random.sample([1,1,1,2], 1)[0]
            new_soln = mutate_path(soln["path"], num_mutations=num_mutations)
            new_score = new_soln["total_score"]
            
            if new_score > score:
                improvement_deltas.append(new_score-score)
                #print("{:5} improved score for {}: {:.0f} --> {:.0f}".format(i, idx, score, new_score))
                population.insert(idx, new_soln)
            else:
                improvement_deltas.append(0)
                population.insert(idx, soln)
            
            if new_score > best_score:
                print("{:4} new best: (usr:{}) {:.0f} --> {:.0f}".format(i, idx, best_score, new_score))
                best_solns.append(new_soln)
                best_score = new_score
                
        except Exception as e:
            print("Error occurred when generating mutation: {}".format(e))
    return sorted(population, key=lambda x:-x["total_score"])
    
def write_solution(soln):
    toy_ids = {t:sorted([x for x in toys[t]["ids"]], reverse=True) for t in list(toys)}

    bags_ids = []
    for step in soln["path"]:
        config = step["config"]
        for _ in range(step["num_bags"]):
            bag_toy_ids = []
            for toy in config:
                if len(toy_ids[toy]) == 0:
                    raise Exception("toy count error! no '{}s' left!".format(toy))
    
                bag_toy_ids.append(toy_ids[toy].pop())
            bags_ids.append(bag_toy_ids)
            
    submit_df = pd.DataFrame({"Gifts": [" ".join(b) for b in bags_ids]})
    submit_df.to_csv("./solution.csv", sep=",", index=False)

#Load toys
gifts, toys = load_gifts_and_toys()
# performance enhancement - randomly select a shuffled index instead of shuffling an array
random_indexes = [shuf(list(range(NUM_SAMPLES))) for _ in range(500)]
cache_tree = generate_cache_tree(toys, num_samples=NUM_SAMPLES, max_depth=3)
main()

