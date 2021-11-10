# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 22:44:12 2021

@author: Kamel Hanna
"""
adj_list={'A':['B', 'C'], 'B':['D', 'E'], 'C':['B', 'E'], 'D':[], 'E':['F'], 'F':[]}
color={}
parent={}
traversal_time={}
dfs_traversal_output=[]
for node in adj_list.keys():
    color[node]='W'
    parent[node]=None
    traversal_time[node]=[-1, -1]
print(color, parent, traversal_time)
time=0
def dfs(u):
    global time
    color[u]='G'
    traversal_time[u][0]=time
    dfs_traversal_output.append(u)
    for v in adj_list[u]:
        if color[v]=='W':
            parent[v]=u
            dfs(v)
    color[u]='B'
    traversal_time[u][1]=time
    time+=1
dfs('A')
for node in adj_list.keys():
    print(node,"->", traversal_time[node])