import numpy as np
from scipy.spatial import Voronoi
from smath.collision import collide_line
from pathplan.graph import Graph

def to_voronoi(graph: Graph, valid_map: np.ndarray) -> (Graph, list):
    vor = Voronoi(np.array(graph.vertices))

    center = vor.points.mean(axis=0)
    ptp_bound = vor.points.ptp(axis=0)
    vw, vh = valid_map.shape
    
    v_idx = 0
    finite_vertices = []
    finite_edges = []
    for point_idxs, vertex_idxs in zip(vor.ridge_points, vor.ridge_vertices):
        vertex_idxs = np.asarray(vertex_idxs) 
        if np.all(vertex_idxs >= 0):
            v1, v2 = vor.vertices[vertex_idxs]
        else:
            i = vertex_idxs[vertex_idxs >= 0][0]  # finite end Voronoi vertex

            t = vor.points[point_idxs[1]] - vor.points[point_idxs[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[point_idxs].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            direction /= np.linalg.norm(direction)
            direction = -direction
            far_point = vor.vertices[i].copy()
            while not(0 <= far_point[0] and far_point[0] < vw and 0 <= far_point[1] and far_point[1] < vh):
                far_point += direction
            far_point -= direction

            v1, v2 = (vor.vertices[i], far_point)
            continue
            
        finite_vertices.append(v1)
        finite_vertices.append(v2)
        finite_edges.append([v_idx, v_idx + 1])
        v_idx += 2
    
    #return Graph(finite_vertices, finite_edges), []

    v_idx = 0
    valid_vertices = []
    valid_edges = []
    target_indices = []
    for finite_idxs in finite_edges:
        v1 = finite_vertices[finite_idxs[0]]
        v2 = finite_vertices[finite_idxs[1]]
        vp1 = (0 <= v1[0] and v1[0] < vw and 0 <= v1[1] and v1[1] < vh)
        vp2 = (0 <= v2[0] and v2[0] < vw and 0 <= v2[1] and v2[1] < vh)
        if (not vp1) and (not vp2):
            continue
        if not vp1:
            dir = v1 - v2
            dir = dir / np.linalg.norm(dir)
            v1 = v2 + dir
            while 0 <= v1[0] and v1[0] < vw and 0 <= v1[1] and v1[1] < vh:
                v1 += dir
            v1 -= dir
        elif not vp2:
            dir = v2 - v1
            dir = dir / np.linalg.norm(dir)
            v2 = v1 + dir
            while 0 <= v2[0] and v2[0] < vw and 0 <= v2[1] and v2[1] < vh:
                v2 += dir
            v2 -= dir
        is_coll = False
        for point_idxs in graph.edges:
            p1 = graph.vertices[point_idxs[0]]
            p2 = graph.vertices[point_idxs[1]]
            is_coll = collide_line((v1,v2),(p1,p2))
            if is_coll:
                break

        if not is_coll:
            def check_v(pos):
                EPS = 0.0
                sx, sy = int(pos[0] - EPS), int(pos[1] - EPS)
                ex, ey = int(pos[0] + EPS), int(pos[1] + EPS)
                done = True
                sx = sx if sx >= 0 else 0
                sy = sy if sy >= 0 else 0
                ex = ex if ex < vw else vw - 1
                ey = ey if ey < vh else vh - 1
                for x in range(sx, ex+1):
                    for y in range(sy, ey+1):
                        done = done and valid_map[x,y]
                return done
            is_valid1 = check_v(v1)
            is_valid2 = check_v(v2)
            if is_valid1 or is_valid2:
                valid_vertices.append(v1)
                valid_vertices.append(v2)
                valid_edges.append([v_idx, v_idx + 1])
                v_idx += 2

                if (is_valid1) and (not is_valid2):
                    target_indices.append(v_idx - 1)    # Append v2's idx
                elif (not is_valid1) and (is_valid2):
                    target_indices.append(v_idx - 2)        # Append v1's idx

    return Graph(valid_vertices, valid_edges), target_indices


    
