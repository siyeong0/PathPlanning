
def collide_line(l1, l2):
    pt1, pt2 = l1
    x1, y1 = pt1
    x2, y2 = pt2
    pt3, pt4 = l2
    x3, y3 = pt3
    x4, y4 = pt4
    uA = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
    uB = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))

    EPS = 0.0
    return uA >= 0.0 - EPS and uA <= 1.0 + EPS and uB >= 0.0 - EPS and uB <= 1.0 + EPS