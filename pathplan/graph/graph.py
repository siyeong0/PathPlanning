import numpy as np
import cv2 as cv

class Graph:
    def __init__(self, vertices : list, edges : list):
        self.vertices: list = None
        self.edges: list = None
        self.neighbors: list = None
        self.initialize(vertices, edges)

    def initialize(self, vertices : list, edges : list):
        self.vertices = vertices
        self.edges = edges
        self.neighbors = []
        for _ in range(len(vertices)):
            self.neighbors.append([])
        
        for edge in self.edges:
            idx1, idx2 = edge
            self.neighbors[idx1].append(idx2)
            self.neighbors[idx2].append(idx1)


