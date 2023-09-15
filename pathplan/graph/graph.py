import numpy as np
import cv2 as cv

class Graph:
    def __init__(self, vertices:list, edges:list):
        self.vertices: list = None
        self.edges: list = None
        self.neighbors: list = None
        self.initialize(vertices, edges)

    def initialize(self, vertices:list, edges:list):
        self.vertices = vertices
        self.edges = edges
        self.neighbors = []
        for _ in range(len(vertices)):
            self.neighbors.append([])
        
        for edge in self.edges:
            idx1, idx2 = edge
            self.neighbors[idx1].append(idx2)
            self.neighbors[idx2].append(idx1)

    def draw(self, dst:np.ndarray, ver_color=(92,255,0), edge_color=(0,255,0)):
        buffer = dst.copy()
        # Draw graph
        for edge in self.edges:  # Draw edges
            v1, v2 = self.vertices[edge[0]], self.vertices[edge[1]]
            buffer = cv.line(buffer, (int(v1[1]), int(v1[0])), (int(v2[1]), int(v2[0])), edge_color, 1)
        for v in self.vertices:  # Draw vertices
            v = (int(v[0]), int(v[1]))
            buffer = cv.rectangle(buffer, (v[1]-2, v[0]-2), (v[1]+2, v[0]+2), ver_color, cv.FILLED)

        return buffer


