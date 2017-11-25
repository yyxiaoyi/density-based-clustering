# -*- coding: utf-8 -*-

import numpy as np

def regionQuery(pid, eps, dis_matrix):
    '''
        Return all points within p's eps-neighborhood(not including p).
        Params:
            pid             point p
            eps             neighborhood
            dis_matrix      pairwise distance matrix
    '''
    neighbors = []
    for nid in range(len(dis_matrix)):
        if nid != pid and dis_matrix[pid][nid] <= eps:
            neighbors.append(nid)
    return neighbors 

def DBSCAN(dis_matrix, eps, mpts):
    '''
        Run DBSCAN with input dataset and parameters.
        params:
            dis_matrix      pairwise distance matrix
            eps             neighborhood of each point
            mpts            min neighbor points        
    '''
    dm = np.asmatrix(dis_matrix)
    m, n = dm.shape
    if m != n:
        raise ValueError('Matrix size dosen\'t match(%d, %d). ' % (m, n))
    if m < 20:
        raise ValueError('Too few points(%d).' % dm.shape(0))
          
    visited = [0]*m
    current_cluster = 1
    
    for pid in range(m):
        if visited[pid] == 0: # p is unvisited
            neighbors = regionQuery(pid, eps, dis_matrix)
            
            if len(neighbors) < mpts:
                visited[pid] = -1 # mark p as noise
            else:
                visited[pid] = current_cluster # p is core point
                
                # expand this cluster
                while len(neighbors) > 0:
                    nid = neighbors[0]
                    neighbors = neighbors[1:]
                    if visited[nid] <= 0:
                        visited[nid] = current_cluster # reachable point
                    
                        n_neighbors = regionQuery(nid, eps, dis_matrix)
                        if len(n_neighbors) >= mpts:
                             neighbors += [i for i in n_neighbors if \
                                           visited[i] <= 0 and \
                                           i not in neighbors]                             
                             
                current_cluster += 1 # move to next cluster
    
    return visited


