import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.nan)
import re
import decimal
import itertools
import copy
from operator import itemgetter
import time
###########################
def float_to_str(f):
    ctx = decimal.Context()
    ctx.prec = 15
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')
###########################
# Dataset 1 (given dataset, graphs*6)
txt_name_list = ['../dataset/graph_{}.txt'.format(i+1) for i in range(6)]

# Dataset2 : IBM Quest Data Generator dataset (w/ [nitems_0.1, ntrans_0.1])
# (directed & bi-directed, graphs*2)
# The following commented codes are for writing original transactions to .txt
# Already pre-do this, so comment them out.
'''
# Write original transactions data to .txt
data_name = '../dataset/ibm/data.nitems_{}.ntrans_{}'.format(0.1, 0.1)
for d in ['directed', 'bidirected']:
    ibm_data = open(data_name, 'r')
    counter = 0
    with open('../dataset/ibm_'+data_name.split('/')[-1]+'_'+d+'.txt', 'w') as f:
        for line in ibm_data.readlines():
            item = re.split(r'[\s:]+',line.strip(' ').strip('\n'))
            if counter == 0:
                trans_bag = [[int(item[0]), int(item[-1])]]
                if d == 'bidirected':
                    trans_bag.append([int(item[-1]), int(item[0])])
            else:
                trans_bag.append([int(item[0]), int(item[-1])])
                if d == 'bidirected':
                    trans_bag.append([int(item[-1]), int(item[0])])
            counter += 1
        trans_bag = sorted(trans_bag, key=lambda x: x[0])
        for i in range(len(trans_bag)):
            f.write(str(trans_bag[i][0])+','+str(trans_bag[i][-1])+'\n')
'''
# Graph 7 (directed)
txt_name_list.append('../dataset/ibm_data.nitems_{}.ntrans_{}_directed.txt'.format(0.1, 0.1))
# Graph 8 (bi-directed)
txt_name_list.append('../dataset/ibm_data.nitems_{}.ntrans_{}_bidirected.txt'.format(0.1, 0.1))
###########################
trans_list = []
# Read transactions of .txt files
for i in range(0, len(txt_name_list)):
    trans_dict = {}
    start_flag = True
    ibm_data = open(txt_name_list[i], 'r')
    start_int = 0
    for line in ibm_data.readlines():
        item = line.split(',')
        item[0] = int(item[0])
        item[1] = int(item[1].strip('\n'))
        if item[0] in trans_dict:
            # append the new end node to the existing start node
            trans_dict[item[0]].append(item[1])
        else:
            # create new start node
            trans_dict[item[0]] = [item[1]]
    # Add nodes that didn't direct to other nodes (never being a start node)
    add_keys = []
    for j in trans_dict:
        for k in trans_dict[j]:
            if k not in trans_dict: add_keys.append(k)
    for k in add_keys: trans_dict[k] = []
    trans_list.append(trans_dict)
#print([len(trans_list[i]) for i in range(len(trans_list))])
# [6, 6, 5, 8, 469, 1228, 100, 100] -> max_num+1 of each graph
# [5, 5, 6, 18, 1102, 5220, 967, 1934] -> num of link edges
# [5, 5, 4, 7, 118, 187, 99, 100] -> num of start nodes
# [6, 5, 4, 7, 469, 1228, 100, 100] -> num of nodes
###########################
def init_adjacency_dict(trans):
    A_mat = trans
    At_mat = {}
    for t in trans: At_mat[t] = []
    for t in trans:
        for link in trans[t]: At_mat[link].append(t)
    return A_mat, At_mat

def init_adjacency_matrix(trans):
    N = len(trans)
    key_ref = [k for k in trans.keys()]
    A_mat = np.zeros((N, N))
    for k, v in trans.items():
        k_ref = key_ref.index(k)
        for vv in v:
            vv_ref = key_ref.index(vv)
            A_mat[k_ref][vv_ref] = 1 / len(trans[k])
    
    return A_mat, A_mat.T

def cal_multiply(matrix, vector):
    res = {}
    for row in matrix:
        res[row] = 0
        for item in matrix[row]: res[row] += vector[item]
    return res

def normalize_dict(dictionary):
    # Normalize the values of a dict to sum up to 1
    norm = sum((dictionary[p] for p in dictionary))
    return {k: v / norm for (k, v) in dictionary.items()}

def diff_sum(v1, v2):
    if not (v1 and v2): return float('inf')
    total = sum([abs(v1[i] - v2[i]) for i in v1])
    return total
###########################
def hits(links, e=1e-4):
    (A_dict, At_dict) = init_adjacency_dict(links)
    # Initialize authority & hub with 1
    authority = dict.fromkeys(links.keys(), 1.0)
    hub = dict(zip(links.keys(), [1]*len(links)))
    authority_prev = None
    hub_prev = None
    while (diff_sum(authority_prev, authority) + diff_sum(hub_prev,hub)) >= e:
        authority_prev = authority
        hub_prev = hub
        hub = normalize_dict(cal_multiply(A_dict, authority_prev))
        authority = normalize_dict(cal_multiply(At_dict, hub_prev))
        #print(authority, hub)
    return authority, hub
###########################
def page_rank(links, e=1e-4, d=0.15):
    N = len(links)
    key_ref = [k for k in links.keys()]
    (A_mat, At_mat) = init_adjacency_matrix(links)
    N_mat = np.ones((N, N)) / N
    R = np.ones((N, 1)) / N
    
    distance = float('inf')
    while distance >= e:
        R_prev = R
        P = d * N_mat + (1 - d) * At_mat
        R = np.dot(P, R_prev)
        # Normalize the values of R to sum up to 1
        R = R / sum(R)
        # Calculate the Euclidean distance of R, R_prev
        distance = np.linalg.norm(R - R_prev)
    
    # Transfer matrix back to dictionary
    links_dict = dict.fromkeys(links.keys(), 0.0)
    for i in range(len(R)): links_dict[key_ref[i]] = float(R[i])
    return links_dict
###########################
def sim_rank(links, max_iter=10, e=1e-4, C=0.8):
    ref = [k for k in links.keys()]
    (A_dict, At_dict) = init_adjacency_dict(links)
    N = len(links)
    sim_prev = np.zeros(N)
    sim = np.identity(N)
    for i in range(max_iter):
        if np.allclose(sim, sim_prev, atol=e): break
        sim_prev = np.copy(sim)
        for a, b in itertools.product(A_dict, A_dict):
            if a is b: continue
            Ia = At_dict[a]
            Ib = At_dict[b]
            if not (Ia and Ib): # Ia or Ib (or both) is empty
                sim[ref.index(a)][ref.index(b)] = 0
                continue
            sum_ab = sum(sim_prev[ref.index(aa)][ref.index(bb)] for aa, bb in itertools.product(Ia, Ib))
            # Set sim(a, b)
            sim[ref.index(a)][ref.index(b)] = C / (len(Ia) * len(Ib)) * sum_ab

    return sim
###########################
time_cost = np.zeros((3, len(trans_list)))
### run HITS
print('@HITS')
for i in range(len(trans_list)):
    startTime = time.time()
    links = trans_list[i]
    (authority, hub) = hits(links)
    print('-------------------------------------------------')
    print('Graph {}'.format(i+1))
    print('\t>> Authority:')
    for k, v in sorted(authority.items()): print('\t\t{}:\t{}'.format(k, float_to_str(v)))
    print('\t>> Hub:')
    for k, v in sorted(hub.items()): print('\t\t{}:\t{}'.format(k, float_to_str(v)))
    # Save Time Cost
    time_cost[0][i] = time.time() - startTime
    
    # Write to .txt file, Authority & Hub are written by descending values.
    with open('../results/graph_{}_hits.txt'.format(i+1), 'w') as f:
        f.write('>> Authority:\n')
        for k, v in sorted(authority.items(), key=itemgetter(1), reverse=True): f.write('\t{}:\t{}\n'.format(k, float_to_str(v)))
        f.write('\n>> Hub:\n')
        for k, v in sorted(hub.items(), key=itemgetter(1), reverse=True): f.write('\t{}:\t{}\n'.format(k, float_to_str(v)))
print('=================================================')
###########################
### run PageRank
print('@PageRank')
for i in range(len(trans_list)):
    startTime = time.time()
    links = trans_list[i]
    pr = page_rank(links)
    pr_list = sorted([(k, float_to_str(v)) for k, v in pr.items()], key=lambda x: (x[1], x[0]), reverse=True)
    print('PageRank of Graph {}: (Sorted by the PageRank value)'.format(i+1))
    for k, v in pr_list: print('\tPage ID {}:\t{}'.format(k, v))
    if i == len(trans_list)-1: print('=================================================')
    else: print('-------------------------------------------------')
    # Save Time Cost
    time_cost[1][i] = time.time() - startTime
    
    # Write to .txt file
    with open('../results/graph_{}_page_rank.txt'.format(i+1), 'w') as f:
        f.write('PageRank of Graph {}: (Sorted by the PageRank value)\n'.format(i+1))
        for k, v in pr_list: f.write('Page ID {}:\t{}\n'.format(k, v))
###########################
### run SimRank
print('@SimRank')
for i in range(5): # only first 5 graphs of given datasets
    startTime = time.time()
    links = trans_list[i]
    # Graph 5 will need more time to calculate the values ...
    sr = sim_rank(links)
    print('SimRank Matrix of Graph {}:{}\n{}'.format(i+1, np.array(sr).shape, sr))
    if i == 4: print('=================================================')
    else: print('-------------------------------------------------')
    # Save Time Cost
    time_cost[2][i] = time.time() - startTime
    
    # Write to .txt file
    with open('../results/graph_{}_sim_rank.txt'.format(i+1), 'w') as f:
        f.write('SimRank Matrix of Graph {}:{}\n{}'.format(i+1, np.array(sr).shape, sr))
###########################
# Find a way (e.g., add/delete some links) to increase hub, authority,
# and PageRank of Node 1 in first 3 graphs respectively.
add_dict = [{'1': [4], '4': [1]}, {'1': [4], '4': [1]}, {'1': [4], '4': [1]}]
for i in range(3): # only first 3 graphs of given datasets
    links_origin = copy.deepcopy(trans_list[i])
    links = copy.deepcopy(trans_list[i])
    # Edit Nodes & Links
    for k, v in add_dict[i].items():
        k = int(k)
        if k in links.keys():
            for vv in v: links[k].append(vv)
        else: links[k] = v

    # Write to .txt file
    with open('../results/graph_{}_node1_improved.txt'.format(i+1), 'w') as f:
        # HITS
        for l, text in zip([links_origin, links], ['Before', 'After']):
            (authority, hub) = hits(l)
            print('-------------------------------------------------')
            print('\t{}'.format(l))
            print('({}) Graph {}'.format(text, i+1))
            print('\t>> Authority:')
            for k, v in authority.items(): print('\t\t{}:\t{}'.format(k, float_to_str(v)))
            print('\t>> Hub:')
            for k, v in hub.items(): print('\t\t{}:\t{}'.format(k, float_to_str(v)))
            
            f.write('\t{}\n'.format(l))
            f.write('({}) Graph {}\n'.format(text, i+1))
            f.write('>> Authority:\n')
            for k, v in sorted(authority.items(), key=itemgetter(1), reverse=True): f.write('\t{}:\t{}\n'.format(k, float_to_str(v)))
            f.write('\n>> Hub:\n')
            for k, v in sorted(hub.items(), key=itemgetter(1), reverse=True): f.write('\t{}:\t{}\n'.format(k, float_to_str(v)))
            f.write('-------------------------------------------------\n')
        f.write('=================================================\n')
        print('=================================================')
        # PageRank
        for l, text in zip([links_origin, links], ['Before', 'After']):
            print('\t{}'.format(l))
            pr = page_rank(l)
            pr_list = sorted([(k, float_to_str(v)) for k, v in pr.items()], key=lambda x: x[1], reverse=True)
            print('({}) PageRank of Graph {}: (Sorted by the PageRank value)'.format(text, i+1))
            for k, v in pr_list: print('\tPage ID{}:\t{}'.format(k, v))
            print('-------------------------------------------------')
            
            f.write('\t{}\n'.format(l))
            f.write('({}) PageRank of Graph {}: (Sorted by the PageRank value)\n'.format(text, i+1))
            for k, v in pr_list: f.write('\tPage ID{}:\t{}\n'.format(k, v))
            f.write('-------------------------------------------------\n')
    print('#################################################')
###########################
# Print Time Cost
title = ['HITS', 'PageRank', 'SimRank']
for i in range(len(time_cost)):
    for j in range(len(time_cost[i])):
        print('Time cost of {}, G{}:\t{:.7f} s'.format(title[i], j+1, time_cost[i][j]))
    if i == len(time_cost)-1:
        print('(Mean) Time cost of {}:\t{:.7f} s\n'.format(title[i], sum(time_cost[i]) / 5))
    else:
        print('(Mean) Time cost of {}:\t{:.7f} s\n'.format(title[i], sum(time_cost[i]) / len(time_cost[i])))
###########################


