import numpy as np
import matplotlib.pylab as plt
from copy import deepcopy

# rng = np.random.default_rng(1)
# for new random simulations
rn = np.random.randint(100, size=1)[0]
rng = np.random.default_rng(rn)

from Hypergraphs import CompleteHypergraph
from Hypergraphs import RandomHypergraph

import pandas as pd
import time

def gillespie_sim(g, beta1, beta2, mu, initial_infections, time_max):
    r"""
    Gillespie algorithm to simulate SIS dynamics on a hypergraph g.
    Outputs: X_t = [time, time_to_event, total_infected, event_type, total_pw, total_ho]
    """
    g_sim = deepcopy(g)

    # initialize all nodes as susceptible `0` or infected `1`
    states = np.zeros(g_sim.number_of_nodes(), dtype=int)
    states[initial_infections] = 1    
    for i, state in enumerate(states):
        g_sim.nodes[i]["state"] = state

    # initial total number of infected nodes, initial `time``, and initial `total_rate`
    total_rate, total_pw, total_ho = initialize_total_rate(g_sim, beta1, beta2, mu)
    total_infected = sum(states)
    time = 0
    event_type = None # initially
    time_to_event = None # initially
    # append output to X_t
    X_t = [[time, time_to_event, total_infected, event_type, total_pw, total_ho]]

    # draw events until time_max or until total_rate drops to 0
    while time < time_max:
        # if total_rate is 0, no more events are possible, end simulation
        if np.isclose(total_rate, 0.):
            print(f"break: total_rate is close to 0")
            # append output to X_t
            X_t.append([time_max, None, total_infected, None, None, None])
            break

        # draw next event
        node_i, time_to_event, event_type = draw_next_event(total_rate, g_sim, beta1, beta2)
        time += time_to_event

        # update states
        total_infected, total_rate, total_pw, total_ho = update_states(
            g_sim, total_infected, node_i, total_rate, total_pw, total_ho, beta1, beta2, mu)
        
        if total_infected == 0:
            print(f"break: total_infected == 0: {total_infected}")
            # append output to X_t
            time += time_to_event
            X_t.append([time, time_to_event, total_infected, event_type, None, None])
            break
        
        # TODO: assert equal: total_rate, sum(node["rate"] for node in g_sim.nodes.values())
        total_rate = sum(node["rate"] for node in g_sim.nodes.values())
        # append output to X_t
        X_t.append([time, time_to_event, total_infected, event_type, total_pw, total_ho])
    
    # on exit append to X_t
    print(f"exited while loop: time={time}, time_to_event={time_to_event}")
    X_t.append([time_max, time_to_event, total_infected, None, None, None])

    return np.array(X_t).transpose()

def initialize_total_rate(g_sim, beta1, beta2, mu):
    total_rate = 0
    # total numbers of S-I and I-S-I relations across all susceptible! nodes
    total_pw, total_ho = 0, 0
    for node_i in range(g_sim.number_of_nodes()):
        # if susceptible, set total rate for infection
        if g_sim.nodes[node_i]["state"] == 0:
            # for pairwise infection, num(nbs(i)) * beta1
            nbs_pw = g_sim.neighbors(node_i, 1)
            inf_pw = sum(1 for node_j in nbs_pw \
                         if g_sim.nodes[node_j]["state"] == 1)
            
            # for hyperedge HO infection, only if both neighbors are infected
            nbs_ho = g_sim.neighbors(node_i, 2)
            inf_ho = sum(1 for node_j, node_k in nbs_ho if \
                         g_sim.nodes[node_j]["state"] == 1 and \
                         g_sim.nodes[node_k]["state"] == 1)
            
            # total infection rate for node i
            rate_i = beta1 * inf_pw + beta2 * inf_ho

            # set node rate
            g_sim.nodes[node_i]["rate"] = rate_i

            # add to total rate
            total_rate += rate_i

            # add to total rates of fire
            total_pw += inf_pw
            total_ho += inf_ho
            
        # if infected, set recovery rate
        elif g_sim.nodes[node_i]["state"] == 1:
            g_sim.nodes[node_i]["rate"] = mu
            total_rate += mu
    
    return total_rate, total_pw, total_ho

def draw_next_event(total_rate, g_sim, beta1, beta2):
    r"""
    Returns the selected `node_i` and waiting time `time_to_event` until the next event. 
    """
    u1, u2, u3 = rng.random(3)
    time_to_event = -np.log(1. - u1) / total_rate
    # TODO: could just set `-np.log(u1) / total_rate` as `u1` already in (0, 1)
    
    # find the selected node_i using linear search
    target_sum = u2 * total_rate
    sum_i = 0
    for i in range(g_sim.number_of_nodes()):
        sum_i += g_sim.nodes[i]["rate"]
        if sum_i >= target_sum:
            break
    node_i = i

    # determine the type of event, knowing event happened for node_i
    event_type = None
    if g_sim.nodes[node_i]["state"] == 0:
        # node_i was susceptible, infection (S --> I) event
        # to determine if it was PW or HO infection:
        # TODO: duplicated code ~ see @`initialize_total_rate()`      
        nbs_pw = g_sim.neighbors(node_i, 1)
        inf_pw = sum(1 for node_j in nbs_pw if g_sim.nodes[node_j]["state"] == 1)

        nbs_ho = g_sim.neighbors(node_i, 2)
        inf_ho = sum(1 for node_j, node_k in nbs_ho if g_sim.nodes[node_j]["state"] == 1 and  \
                     g_sim.nodes[node_k]["state"] == 1)

        rate_pw = beta1 * inf_pw
        rate_ho = beta2 * inf_ho

        # first check if infection is possible and what kind
        if rate_pw + rate_ho == 0:
            event_type = "RC"
        elif rate_ho == 0:
            event_type = "PW"
        else:
            # as for selected node_i, random threshold to determine the type of infection
            if u3 < rate_pw / (rate_pw + rate_ho):
                event_type = "PW"
            else:
                event_type = "HO"
    else:
        # recovery (I --> S) event
        event_type = "RC"

    return node_i, time_to_event, event_type

def update_states(g_sim, total_infected, node_i, total_rate, total_pw, total_ho, beta1, beta2, mu):
    r"""
    Updates nodes states and rates based on selected `node_i`. 
    """
    state_before = g_sim.nodes[node_i]["state"]
    if state_before == 0:
        # it doesn"t matter, how it became infected, we need to update both PW and HO rates
        # update count of infected and node i state
        total_infected += 1
        g_sim.nodes[node_i]["state"] = np.int64(1)

        # update total rate and set recovery rate for node i
        total_rate += -g_sim.nodes[node_i]["rate"] + mu
        g_sim.nodes[node_i]["rate"] = mu

        # update rates of susceptible pair-wise neighbors, total rate, and total_pw
        for node_j in g_sim.neighbors(node_i, 1):
            if g_sim.nodes[node_j]["state"] == 0:
                g_sim.nodes[node_j]["rate"] += beta1
                total_rate += beta1
                total_pw += 1
            # update total_pw count since node_i is no longer susceptible
            elif g_sim.nodes[node_j]["state"] == 1:
                total_pw -= 1
        
        # update rates of susceptible higher-order neighbors, total rate, and total_ho
        # TODO: is this okay:
        # Node `i` was suscepible and it became infected through higher order infection 
        # Let `K = (i, j, k)` be hyperedge in g_sim.
        # Then, for this pair of higher-order neighbors `(j, k)` of `i`:
        # - increase the rate of `k` for beta2, only if `k` is susceptible and `j` is infected
        # - increase the rate of `j` for beta2, if `j` is susceptible and `k` is infected
        for node_j, node_k in g_sim.neighbors(node_i, 2):
            if g_sim.nodes[node_j]["state"] == 1 and g_sim.nodes[node_k]["state"] == 0:
                g_sim.nodes[node_k]["rate"] += beta2
                total_rate += beta2
                total_ho += 1
            
            if g_sim.nodes[node_k]["state"] == 1 and g_sim.nodes[node_j]["state"] == 0:
                g_sim.nodes[node_j]["rate"] += beta2
                total_rate += beta2
                total_ho += 1

            # decrease the total_ho if HO nbs (j, k) of node_i are both infected!
            if g_sim.nodes[node_k]["state"] == 1 and g_sim.nodes[node_j]["state"] == 1:
                total_rate -= beta2
                total_ho -= 1
    
    else:
        # recovery event "RC"
        total_infected -= 1
        total_rate -= mu
        g_sim.nodes[node_i]["state"] = np.int64(0)

        g_sim.nodes[node_i]["rate"] = 0. # reset node i rate to 0
        # calculate rate of node_i and update total_rate
        # TODO: duplicated code ~ see @`initialize_total_rate()`
        # for pairwise infection, num(nbs(i)) * beta1
        nbs_pw = g_sim.neighbors(node_i, 1)
        inf_pw = sum(1 for node_j in nbs_pw \
                        if g_sim.nodes[node_j]["state"] == 1)
        
        # for hyperedge HO infection, only if both neighbors are infected
        nbs_ho = g_sim.neighbors(node_i, 2)
        inf_ho = sum(1 for node_j, node_k in nbs_ho if \
                        g_sim.nodes[node_j]["state"] == 1 and \
                        g_sim.nodes[node_k]["state"] == 1)
        
        # infection rate for node_i
        rate_i = beta1 * inf_pw + beta2 * inf_ho

        if rate_i > 0:
            # set infection rate of node_i, add to total rate, 
            # and update total_pw, total_ho
            g_sim.nodes[node_i]["rate"] = rate_i
            total_rate += rate_i
            total_pw += inf_pw
            total_ho += inf_ho
        
        # update rates of susceptible pairwise neighbors, total_rate, and total_pw 
        for node_j in g_sim.neighbors(node_i, 1):
            if g_sim.nodes[node_j]["state"] == 0:
                g_sim.nodes[node_j]["rate"] -= beta1
                total_rate -= beta1
                total_pw -= 1
        
        # update rates of susceptible higher-order neighbors, total_rate, and total_ho
        # Node `i` recovered and is now susceptible, 
        #   for each pair of higher-order neighbors `(j, k)` of `i`: 
        #     decreases by beta2 only susceptible higher-order neighbors:
        #        if `j` is infected and `k` is susceptible, decrease rate of `k`
        #        if `k` is infected and `j` is susceptible, decrease rate of `j`        
        for node_j, node_k in g_sim.neighbors(node_i, 2):
            if g_sim.nodes[node_j]["state"] == 1 and g_sim.nodes[node_k]["state"] == 0:
                g_sim.nodes[node_k]["rate"] -= beta2
                total_rate -= beta2
                total_ho -= 1

            if g_sim.nodes[node_k]["state"] == 1 and g_sim.nodes[node_j]["state"] == 0:
                g_sim.nodes[node_j]["rate"] -= beta2
                total_rate -= beta2
                total_ho -= 1              
    
    return total_infected, total_rate, total_pw, total_ho

def get_average(X_sims, time_max, nsims, delta_t=0.1, selected=2):
    r"""
    Returns average numbers of infected over time.
    """
    # selected is always 1, number of infected 
    times = np.arange(0, time_max + delta_t, delta_t)
    avg_nums = np.zeros(len(times))
    for X_t in X_sims:
        j = 0
        # TODO: try including curves that die out
        # avg_nums[i] += X_t[j] is just adding zeros then to the bins
        # so we can ignore those bins, and break out of the linear search
        for i, time in enumerate(times):
            avg_nums[i] += X_t[selected][j]
            # move to the next time in times
            while X_t[0][j] < time:
                if j == len(X_t[0]) - 1:
                    break
                j += 1
    return avg_nums / nsims, times

def run_on_random():
    file_name = "../data/sim_random.csv"
    # TODO: test and fix to scale properly
    # TODO: put all setups in `config.py` 

    N = 100
    p1 = 5 / (N) # probability of an pairwise edge for random graphs
    p2 = 5 / (N*N)
    
    nsims = 10

    I0 = 10 # number of initial infected
    time_max = 10 # maximum time duration

    # generate hypergraph
    # g = CompleteHypergraph(N)
    g = RandomHypergraph(N, p1, p2)
    g.print()

    # On random:
    # TODO: if we scale p1, p2 properly, then the expected number of edges per node and hyperedges per node will be order 1
    # then the infection can be just real values (positive) no scaling involving N 
    # TODO: what values of beta1, beta2, mu should we use for random graphs, 1/p times complete?
    beta1 =  0.1 * 0.5  # pairwise infection rate
    beta2 =  0.1 * 1  # hyperedge contagion rate
    mu    =  1                  # recovery rate

    initial_infections = list(range(I0)) # which nodes are infected at t=0

    # simulations
    start_time = time.time()
    X_sims = []
    for _ in range(nsims):
        X_t = gillespie_sim(g, beta1, beta2, mu, initial_infections, time_max)
        X_sims.append(X_t)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed_time: {elapsed_time:.2f} seconds for {nsims} simulations with {N} nodes.")

    # export to CSV
    #  nsim [time, time_to_event, total_infected, event_type, total_pw, total_ho]
    data = []
    for nsim, X_t in enumerate(X_sims, start=1):
        times, times_to_event, infected, event_types, total_pws, total_hos = X_t[0], X_t[1], X_t[2], X_t[3], X_t[4], X_t[5]
        nsim_column = np.full(len(times), nsim)
        data.append(pd.DataFrame({"nsim": nsim_column, 
                                  "time": times, 
                                  "time_to_event": times_to_event,
                                  "infected": infected,
                                  "event_type": event_types,
                                  "total_pw": total_pws,
                                  "total_ho": total_hos,
                                  }))
    df = pd.concat(data, ignore_index=True)
    df.to_csv(file_name, index=False)

def run_on_complete():
    file_name = "../data/sim_complete.csv"
    # setup
    # On complete:
    # TODO: put all setups in `config.py` 
    # the reason for this is that the rates in the gillespie are different from rates in the ODE limit
    # in the ODE we have \tau = \beta_1 / N, and \delta = beta_2 / N^2,
    # e.g. if \tau = 1 and \delta = 2 in the ODE, then in gillespie we need to use \tau / N, and \delta / N^2
    # TODO: check the paper with Iacopo and ..
    # TODO: don"t need to use this
    # beta1 =  2 / N       # pairwise infection rate
    # beta2 =  4 / (N**2)  # hyperedge contagion rate
    # mu    =  1                  # recovery rate    
    ###
    ## Example Setup
    #
    #   H = CompleteHypergraph, N = 100, I0 = 10
    #   beta1 = 0.02, beta2 = 0.0004, mu = 1    
    #
    nsims = 10
    N = 100
    beta1 = 2 / N       # pairwise infection rate
    beta2 = 4 / (N**2)  # hyperedge contagion rate
    mu    = 1           # recovery rate
    I0 = 10         # number of initial infected
    time_max = 10   # maximum time duration
    initial_infections = list(range(I0)) # which nodes are infected at t=0
    g = CompleteHypergraph(N)
    print(f"Setup: \n")
    print(f"\tH = {g.__class__.__name__}, N = {N}, I0 = {I0}\n")
    print(f"\tbeta1 = {beta1}, beta2 = {beta2}, mu = {mu}\n")

    # run gillespie simulations
    start_time = time.time()
    X_sims = []
    for _ in range(nsims):
        X_t = gillespie_sim(g, beta1, beta2, mu, initial_infections, time_max)
        X_sims.append(X_t)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed_time: {elapsed_time:.2f} seconds for {nsims} simulations with {N} nodes")

    # export to CSV
    #  nsim [time, time_to_event, infected, event_type, total_pw, total_ho]
    data = []
    for nsim, X_t in enumerate(X_sims, start=1):
        times, times_to_event, infected, event_types, total_pws, total_hos = \
            X_t[0], X_t[1], X_t[2], X_t[3], X_t[4], X_t[5]
        nsim_column = np.full(len(times), nsim)
        data.append(pd.DataFrame({"nsim": nsim_column, 
                                  "time": times, 
                                  "time_to_event": times_to_event,
                                  "infected": infected,
                                  "event_type": event_types,
                                  "total_pw": total_pws,
                                  "total_ho": total_hos,
                                  }))
    df = pd.concat(data, ignore_index=True)
    df.to_csv(file_name, index=False)


if __name__ == "__main__":
    run_on_complete()
