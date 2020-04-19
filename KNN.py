# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 22:01:31 2020

@author: jzg
"""


from math import sqrt
from collections import namedtuple

from time import clock
from random import random


class KdNode(object):
    def __init__(self,dom_elt,split,left,right):
        self.dom_elt=dom_elt
        self.split=split
        self.left=left
        self.right=right


class KdTree(object):
    def __init__(self,data):
        k=len(data[0])
        
        def CreateNode(split,data_set):
            if not data_set:
                return None
            data_set.sort(key=lambda x: x[split])
            split_pos=len(data_set)//2
            median=data_set[split_pos]
            split_next=(split+1)%k
            
            return KdNode(median,split,
                          CreateNode(split_next,data_set[:split_pos]),
                          CreateNode(split_next,data_set[split_pos+1:]))
        self.root = CreateNode(0,data)
        
        
def preorder(root):
    print(root.dom_elt)
    if root.left:
        preorder(root.left)
    if root.right:
        preorder(root.right)


result=namedtuple("Result_tuple","nearest_point nearest_dist nodes_visited")


def find_nearest(tree,point):
    k=len(point)

    def travel(kd_node,target,max_dist):
        if kd_node is None:
            return result([0]*k,float("inf"),0)
        
        nodes_visited=1
        
        s=kd_node.split
        pivot=kd_node.dom_elt
        
        if target[s] <= pivot[s]:
            nearer_node=kd_node.left
            further_node=kd_node.right
        else:
            nearer_node=kd_node.right
            further_node=kd_node.left
        
        temp1=travel(nearer_node,target,max_dist)
        
        nearest = temp1.nearest_point
        dist=temp1.nearest_dist
        
        nodes_visited+=temp1.nodes_visited
        
        if dist<max_dist:
            max_dist=dist
        
        temp_dist=abs(pivot[s]-target[s])
        if max_dist<temp_dist:
            return result(nearest,dist,nodes_visited)
        
        temp_dist = sqrt(sum((p1-p2)**2 for p1,p2 in zip(pivot,target)))
        
        if temp_dist<dist:
            nearest=pivot
            dist=temp_dist
            max_dist=dist
            
        temp2=travel(further_node,target,max_dist)
        
        nodes_visited+=temp2.nodes_visited
        
        if temp2.nearest_dist < dist:
            nearest=temp2.nearest_point
            dist=temp2.nearest_dist
            
        return result(nearest,dist,nodes_visited)
    
    return travel(tree.root,point,float("inf"))


def random_point(k):
    return [random() for _ in range(k)]

def random_points(k,n):
    return [random_point(k) for _ in range(n)]

data = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
kd = KdTree(data)
preorder(kd.root)

ret=find_nearest(kd,[3,4.5])
print(ret)


N = 400000
t0 = clock()
kd2 = KdTree(random_points(3, N))            # 构建包含四十万个3维空间样本点的kd树
ret2 = find_nearest(kd2, [0.1,0.5,0.8])      # 四十万个样本点中寻找离目标最近的点
t1 = clock()
print ("time: ",t1-t0, "s")
print (ret2)
        
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        