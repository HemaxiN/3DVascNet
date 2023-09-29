#!/bin/env python

##
## calculate the concave hull of a set of points
## see: CONCAVE HULL: A K-NEAREST NEIGHBOURS APPROACH
##      FOR THE COMPUTATION OF THE REGION OCCUPIED BY A
##      SET OF POINTS
##      Adriano Moreira and Maribel Yasmina Santos 2007
##

import numpy as np
import scipy.spatial as spt
import matplotlib
#matplotlib.use('Agg')

#!/bin/env python

##
## determine if two line segments intersect
## see: martin-thoma.com/how-to-check-if-two-line-segments-intersect/
##

import numpy as np

def doBoundingBoxesIntersect(a, b, c, d):
    '''
    Check if bounding boxes do intersect. If one bounding box touches
    the other, they do intersect.
    First segment is of points a and b, second of c and d.
    '''
    ll1_x = min(a[0],b[0]); ll2_x = min(c[0],d[0])
    ll1_y = min(a[1],b[1]); ll2_y = min(c[1],d[1])
    ur1_x = max(a[0],b[0]); ur2_x = max(c[0],d[0])
    ur1_y = max(a[1],b[1]); ur2_y = max(c[1],d[1])

    return ll1_x <= ur2_x and \
           ur1_x >= ll2_x and \
           ll1_y <= ur2_y and \
           ur1_y >= ll2_y

def isPointOnLine(a,b,c):
    '''
    Check if a point is on a line.
    '''
    # move to origin
    aTmp = (0,0)
    bTmp = (b[0] - a[0], b[1] - a[1])
    cTmp = (c[0] - a[0], c[1] - a[1])
    r = np.cross(bTmp, cTmp)
    return np.abs(r) < 0.0000000001

def isPointRightOfLine(a,b,c):
    '''
    Check if a point (c) is right of a line (a-b).
    If (c) is on the line, it is not right it.
    '''
    # move to origin
    aTmp = (0,0)
    bTmp = (b[0] - a[0], b[1] - a[1])
    cTmp = (c[0] - a[0], c[1] - a[1])
    return np.cross(bTmp, cTmp) < 0

def lineSegmentTouchesOrCrossesLine(a,b,c,d):
    '''
    Check if line segment (a-b) touches or crosses
    line segment (c-d).
    '''
    return isPointOnLine(a,b,c) or \
           isPointOnLine(a,b,d) or \
          (isPointRightOfLine(a,b,c) ^
           isPointRightOfLine(a,b,d))

def doLinesIntersect(a,b,c,d):
    '''
    Check if line segments (a-b) and (c-d) intersect.
    '''
    return doBoundingBoxesIntersect(a,b,c,d) and \
           lineSegmentTouchesOrCrossesLine(a,b,c,d) and \
           lineSegmentTouchesOrCrossesLine(c,d,a,b)


##############################
## Tests
##############################

def test_doBoundingBoxesIntersect():
    A=(1,1); B=(2,2); C=(3,1); D=(4,2)
    assert doBoundingBoxesIntersect(A,B,C,D) == False
    A=(1,2); B=(3,4); C=(2,1); D=(4,3)
    assert doBoundingBoxesIntersect(A,B,C,D) == True

def test_isPointOnLine():
    A=(1,1); B=(3,3); C=(2,2)
    assert isPointOnLine(A,B,C) == True
    A=(1,1); B=(3,3); C=(3,2)
    assert isPointOnLine(A,B,C) == False


def test_isPointRightOfLine():
    A=(1,1); B=(3,3); C=(2,2)
    assert isPointRightOfLine(A,B,C) == False
    A=(1,1); B=(3,3); C=(3,2)
    assert isPointRightOfLine(A,B,C) == True
    A=(1,1); B=(3,3); C=(1,2)
    assert isPointRightOfLine(A,B,C) == False

# a lot of testcases to be tested with the final function

def tcase(name):
    if name == 'F1':
        return (0,0), (7,7), (3,4), (4,5), False
    elif name == 'F2':
        return (-4,4), (-2,1), (-2,3), (0,0), False
    elif name == 'F3':
        return (0,0), (0,1), (2,2), (2,3), False
    elif name == 'F4':
        return (0,0), (0,1), (2,2), (3,2), False
    elif name == 'F5':
        return (-1,-1), (2,2), (3,3), (5,5), False
    elif name == 'F6':
        return (0,0), (1,1), (2,0), (0.5,2), False
    elif name == 'F7':
        return (1,1), (4,1), (2,2), (3,2), False
    elif name == 'F8':
        return (0,5), (6,0), (2,1), (2,2), False
    elif name == 'T1':
        return (0,-2), (0,2), (-2,0), (2,0), True
    elif name == 'T2':
        return (5,5), (0,0), (1,1), (8,2), True
    elif name == 'T3':
        return (-1,0), (0,0), (-1,-1), (-1,1), True
    elif name == 'T4':
        return (0,2), (2,2), (2,0), (2,4), True
    elif name == 'T5':
        return (0,0), (5,5), (1,1), (3,3), True
    elif name == 'T6':
        return (0,0), (3,3), (0,0), (3,3), True

cases = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
         'T1', 'T2', 'T3', 'T4', 'T5', 'T6']

def check_intersection(name):
    A,B,C,D, result = tcase(name)
    assert doLinesIntersect(A,B,C,D) == result

def test_doLinesIntersect():
    for case in cases:
        yield check_intersection, case


def GetFirstPoint(dataset):
    ''' Returns index of first point, which has the lowest y value '''
    # todo: what if there is more than one point with lowest y?
    imin = np.argmin(dataset[:,1])
    return dataset[imin]

def GetNearestNeighbors(dataset, point, k):
    ''' Returns indices of k nearest neighbors of point in dataset'''
    # todo: experiment with leafsize for performance
    mytree = spt.cKDTree(dataset,leafsize=10)
    distances, indices = mytree.query(point,k)
    # todo: something strange here, we get more indices than points in dataset
    #       so have to do this
    return dataset[indices[:dataset.shape[0]]]

def SortByAngle(kNearestPoints, currentPoint, prevPoint):
    ''' Sorts the k nearest points given by angle '''
    angles = np.zeros(kNearestPoints.shape[0])
    i = 0
    for NearestPoint in kNearestPoints:
        # calculate the angle
        angle = np.arctan2(NearestPoint[1]-currentPoint[1],
                NearestPoint[0]-currentPoint[0]) - \
                np.arctan2(prevPoint[1]-currentPoint[1],
                prevPoint[0]-currentPoint[0])
        angle = np.rad2deg(angle)
        # only positive angles
        angle = np.mod(angle+360,360)
        #print NearestPoint[0], NearestPoint[1], angle
        angles[i] = angle
        i=i+1
    return kNearestPoints[np.argsort(angles)]

def plotPoints(dataset):
    matplotlib.pyplot.plot(dataset[:,0],dataset[:,1],'o',markersize=10,markerfacecolor='0.75',
            markeredgewidth=1)
    matplotlib.pyplot.axis('equal')
    matplotlib.pyplot.axis([min(dataset[:,0])-0.5,max(dataset[:,0])+0.5,min(dataset[:,1])-0.5,
        max(dataset[:,1])+0.5])
    matplotlib.pyplot.show()

def plotPath(dataset, path):
    matplotlib.pyplot.plot(dataset[:,0],dataset[:,1],'o',markersize=10,markerfacecolor='0.65',
            markeredgewidth=0)
    path = np.asarray(path)
    matplotlib.pyplot.plot(path[:,0],path[:,1],'o',markersize=10,markerfacecolor='0.55',
            markeredgewidth=0)
    matplotlib.pyplot.plot(path[:,0],path[:,1],'-',lw=1.4,color='k')
    matplotlib.pyplot.axis('equal')
    matplotlib.pyplot.axis([min(dataset[:,0])-0.5,max(dataset[:,0])+0.5,min(dataset[:,1])-0.5,
        max(dataset[:,1])+0.5])
    matplotlib.pyplot.axis('off')
    matplotlib.pyplot.savefig('./doc/figure_1.png', bbox_inches='tight')
    matplotlib.pyplot.show()

def removePoint(dataset, point):
    delmask = [np.logical_or(dataset[:,0]!=point[0],dataset[:,1]!=point[1])]
    newdata = dataset[tuple(delmask)]
    return newdata


def concaveHull(dataset, k):
    assert k >= 3, 'k has to be greater or equal to 3.'
    points = dataset
    # todo: remove duplicate points from dataset
    # todo: check if dataset consists of only 3 or less points
    # todo: make sure that enough points for a given k can be found

    firstpoint = GetFirstPoint(points)
    # init hull as list to easily append stuff
    hull = []
    # add first point to hull
    hull.append(firstpoint)
    # and remove it from dataset
    points = removePoint(points,firstpoint)
    currentPoint = firstpoint
    # set prevPoint to a Point righ of currentpoint (angle=0)
    prevPoint = (currentPoint[0]+10, currentPoint[1])
    step = 2

    while ( (not np.array_equal(firstpoint, currentPoint) or (step==2)) and points.size > 0 ):
        if ( step == 5 ): # we're far enough to close too early
            points = np.append(points, [firstpoint], axis=0)
        kNearestPoints = GetNearestNeighbors(points, currentPoint, k)
        cPoints = SortByAngle(kNearestPoints, currentPoint, prevPoint)
        # avoid intersections: select first candidate that does not intersect any
        # polygon edge
        its = True
        i = 0
        while ( (its==True) and (i<cPoints.shape[0]) ):
                i=i+1
                if ( np.array_equal(cPoints[i-1], firstpoint) ):
                    lastPoint = 1
                else:
                    lastPoint = 0
                j = 2
                its = False
                while ( (its==False) and (j<np.shape(hull)[0]-lastPoint) ):
                    its = doLinesIntersect(hull[step-1-1], cPoints[i-1],
                            hull[step-1-j-1],hull[step-j-1])
                    j=j+1
        if ( its==True ):
            print("all candidates intersect -- restarting with k = ",k+1)
            return concaveHull(dataset,k+1)
        prevPoint = currentPoint
        currentPoint = cPoints[i-1]
        # add current point to hull
        hull.append(currentPoint)
        points = removePoint(points,currentPoint)
        step = step+1
    # check if all points are inside the hull
    p = matplotlib.path.Path(hull)
    pContained = p.contains_points(dataset, radius=0.0000000001)
    if (not pContained.all()):
        print ("not all points of dataset contained in hull -- restarting with k = ",k+1)
        return concaveHull(dataset, k+1)

    print ("finished with k = ",k)
    return hull


############################################################
## tests
############################################################

nx = 5
ny = nx
tpoints = []
tpointsy = []
x = np.arange(nx)
for i in x:
        tpoints.append((i,0))
        tpointsy.append((0,i))
tpoints = np.asarray(tpoints)
tpointsy = np.asarray(tpointsy)

def test_GetNearestNeighbors_1d_x_00():
    neighbors = GetNearestNeighbors(tpoints, (0,0),nx)
    assert np.array_equal(neighbors,tpoints)
def test_GetNearestNeighbors_1d_x_nx():
    neighbors = GetNearestNeighbors(tpoints, (nx,nx),nx)
    assert np.array_equal(neighbors,tpoints[::-1])

def test_GetNearestNeighbors_1d_y_00():
    neighbors = GetNearestNeighbors(tpointsy, (0,0),nx)
    assert np.array_equal(neighbors,tpointsy)
def test_GetNearestNeighbors_1d_y_nx():
    neighbors = GetNearestNeighbors(tpointsy, (nx,nx),nx)
    assert np.array_equal(neighbors,tpointsy[::-1])


################################################
clock = np.array([[3,2], [4,2], [4,3], [4,4], [3,4], [2,4], [2,3], [2,2]])

def check_SortByAngle(points,currentPoint,prevPoint,i):
    sortedPoints = SortByAngle(points,currentPoint,prevPoint)
    #for j in np.arange(len(clock)):
    #    print sortedPoints[j], clock[j], np.roll(clock,-i,axis=0)[j]
    assert np.array_equal(sortedPoints, np.roll(clock,-i,axis=0))

def test_SortByAngle_clock():
    i=0
    for point in clock:
        yield check_SortByAngle, clock, (3,3), point, i
        i=i+1

################################################
# simple test dataset
points = np.array([[10,  9], [ 9, 18], [16, 13], [11, 15], [12, 14], [18, 12],
                   [ 2, 14], [ 6, 18], [ 9,  9], [10,  8], [ 6, 17], [ 5,  3],
                   [13, 19], [ 3, 18], [ 8, 17], [ 9,  7], [ 3,  0], [13, 18],
                   [15,  4], [13, 16]])
points_solution_k_5 = np.array([[3, 0],[10,  8],[15,  4],[18, 12],[13, 18],[13, 19],
                               [ 9, 18],[6, 18],[3, 18],[2, 14],[9, 9],[5, 3],[3, 0]
                               ])
def test_concaveHull_1_k_5():
    hull = concaveHull(points,5)
    assert np.array_equal(hull, points_solution_k_5)

def test_concaveHull_1_k_3():
    # this tests, if missed point (too far away) is detected and if the
    # function is started again with increased k
    hull = concaveHull(points,3)
    assert np.array_equal(hull, points_solution_k_5)


# points to test what happens if all points intersect
points_intersect = np.array([[1,1],[10,3],[11,8],[9,14],[15,21],[-5,15],[-3,10],
                            [2,5],    # from here the distracting points
                            [9,10],[8,9],[8,11],[8,12],[9,11],[9,12]
                            ])
points_intersect_solution = np.array([[1, 1],[10,  3],[11,  8],[9, 14],[15, 21],
                                     [-5, 15],[-3, 10],[1, 1]
                                     ])
def test_concaveHull_intersect():
    hull = concaveHull(points_intersect, 5)
    assert np.array_equal(hull, points_intersect_solution)



points_E = np.array([[1,1],[2,1],[3,1],[4,1],[5,1],[6,1],[6,2],[6,3],[5,3],[4,3],
                    [3,3],[3,4],[3,5],[4,5],[5,5],[5,6],[5,7],[4,7],[3,7],[3,8],
                    [3,9],[4,9],[5,9],[6,9],[6,10],[6,11],[5,11],[4,11],[3,11],[2,11],
                    [1,11],[1,10],[1,9],[1,8],[1,7],[1,6],[1,5],[1,4],[1,3],[1,2],
                    [5,2],[4,2],[3,2],[2,2],[2,3],[2,4],[2,5],[2,6],[2,7],[2,8],
                    [2,9],[2,10],[3,10],[4,10],[5,10],[3,6],[4,6],[5,6],[4.5,7],[3,8.5],
                    ])

