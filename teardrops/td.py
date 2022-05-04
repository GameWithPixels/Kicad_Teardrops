#!/usr/bin/env python

# Teardrop for pcbnew using filled zones
# (c) Niluje 2019 thewireddoesntexist.org
#
# Based on Teardrops for PCBNEW by svofski, 2014 http://sensi.org/~svo
# Cubic Bezier upgrade by mitxela, 2021 mitxela.com

from math import cos, sin, asin, atan2, sqrt, pi
from pcbnew import PCB_VIA, ToMM, PCB_TRACK, PCB_ARC, FromMM, wxPoint, GetBoard, ZONE
from pcbnew import PAD_ATTRIB_PTH, PAD_ATTRIB_SMD, ZONE_FILLER, VECTOR2I
from pcbnew import STARTPOINT, ENDPOINT, ZONE_SETTINGS, ZONE_CONNECTION_FULL, ZONE_FILL_MODE_POLYGONS
from pcbnew import PAD_SHAPE_RECT, PAD_SHAPE_CIRCLE

__version__ = "0.6.0"

ToUnits = ToMM
FromUnits = FromMM

def PointDistance(a, b):
    """Distance between two points"""
    return sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]))

def NormalizeVector(pt):
    """Make vector unit length"""
    norm = sqrt(pt.x * pt.x + pt.y * pt.y)
    if norm > 0:
        return [t / norm for t in pt]
    else:
        return pt

def CrossProduct(pt1, pt2):
    return pt1[0] * pt2[1] - pt1[1] * pt2[0]

def DotProduct(pt1, pt2):
    return pt1[0] * pt2[0] + pt1[1] * pt2[1]

def DrawCross(pt):
    brd = GetBoard()
    ds = PCB_SHAPE(brd)
    ds.SetStart(wxPoint(pt.x - FromMM(0.1), pt.y))
    ds.SetEnd(wxPoint(pt.x + FromMM(0.1), pt.y))
    ds.SetLayer(Dwgs_User)
    ds.SetWidth(FromMM(0.005))
    brd.Add(ds)
    ds = PCB_SHAPE(brd)
    ds.SetStart(wxPoint(pt.x, pt.y - FromMM(0.1)))
    ds.SetEnd(wxPoint(pt.x, pt.y + FromMM(0.1)))
    ds.SetLayer(Dwgs_User)
    ds.SetWidth(FromMM(0.005))
    brd.Add(ds)

def DrawLine(pt1, pt2):
    brd = GetBoard()
    ds = PCB_SHAPE(brd)
    ds.SetStart(pt1)
    ds.SetEnd(pt2)
    ds.SetLayer(Dwgs_User)
    ds.SetWidth(FromMM(0.005))
    brd.Add(ds)

def normalizeAngle(angle):
    while angle < 0:
        angle += 2 * pi
    while angle > 2 * pi:
        angle -= 2 * pi
    return angle     

MAGIC_TEARDROP_ZONE_ID = 0x4242

class Line:
    # Define an infinite line, which we use to find points on the outline of a pad
    def __init__(self, point, tangent) -> None:
        self.point = point
        self.tangent = tangent

class OutlineSegment:
    # Pad outlines are made of segments and 90-degree arcs, this is a segment
    def __init__(self, start, end) -> None:
        self.start = start
        self.end = end
        self.delta = end - start
        self.length = sqrt(self.delta.x * self.delta.x + self.delta.y * self.delta.y)
        self.tangent = [t / self.length for t in self.delta]

    # Find the intersection, if any, between this segment and the line passed in
    def intersect(self, line: Line):
        delta = line.point - self.start
        denom = CrossProduct(self.delta, line.tangent)
        if denom != 0:
            param = CrossProduct(delta, line.tangent) / denom
            if param > 0 and param < 1:
                # the segment and line intersect
                pt = wxPoint(self.start.x + self.delta.x * param, self.start.y + self.delta.y * param)
                return pt, param
            else:
                return None
        else:
            # colinear
            return None

    def getTangent(self, param):
        return self.tangent

    def draw(self):
        DrawLine(self.start, self.end)

class OutlineArc:
    # Pad outlines are made of segments and 90-degree arcs, this is a 90-degree arc
    def __init__(self, center, radius, startAngle) -> None:
        self.center = center
        self.radius = radius
        self.startAngle = startAngle

    # Find the intersection, if any, between this arc and the line passed in
    # Note, this returns the closest intersection to the line's anchor point
    def intersect(self, line: Line):
        lineNormal = [line.tangent[1], -line.tangent[0]] # 90 deg clockwise from tangent
        denom = DotProduct(lineNormal, lineNormal)
        if denom != 0:
            signedDist = DotProduct(line.point - self.center, lineNormal) / sqrt(denom)
            distSqr = signedDist * signedDist
            radiusSqr = self.radius * self.radius
            if distSqr < radiusSqr:
                # intersects circle, compute intersection points and angles
                oppositeEdgeLength = sqrt(radiusSqr - distSqr)
                vec1 = [signedDist * lineNormal[0] + oppositeEdgeLength * line.tangent[0], \
                    signedDist * lineNormal[1] + oppositeEdgeLength * line.tangent[1]]
                vec2 = [signedDist * lineNormal[0] - oppositeEdgeLength * line.tangent[0], \
                    signedDist * lineNormal[1] - oppositeEdgeLength * line.tangent[1]]

                # sort points by closest to the line's anchor point
                point1 = self.center + wxPoint(vec1[0], vec1[1])
                angle1 = atan2(-vec1[1], vec1[0])

                point2 = self.center + wxPoint(vec2[0], vec2[1])
                angle2 = atan2(-vec2[1], vec2[0])

                if PointDistance(line.point, point1) < PointDistance(line.point, point2):
                    pointClosest = point1
                    pointFurthest = point2
                    angleClosest = angle1
                    angleFurthest = angle2
                else:
                    pointClosest = point2
                    pointFurthest = point1
                    angleClosest = angle2
                    angleFurthest = angle1

                # if the closest intersection point actually on the valid arc portion of the circle?
                angleDeltaClosest = normalizeAngle(angleClosest - self.startAngle)
                if angleDeltaClosest >= 0 and angleDeltaClosest <= 0.5 * pi:
                    # return point and param
                    param = angleDeltaClosest / (0.5 * pi)
                    return pointClosest, param
                else:
                    # how about the furthest point?
                    angleDeltaFurthest = normalizeAngle(angleFurthest - self.startAngle)
                    if angleDeltaFurthest >= 0 and angleDeltaFurthest <= 0.5 * pi:
                        # return point and param
                        param = angleDeltaFurthest / (0.5 * pi)
                        return pointFurthest, param
                    else:
                        return None
            else:
                return None
        else:
            return None

    def getPoint(self, param):
        paramAngle = self.startAngle + param * 0.5 * pi
        return self.center + wxPoint(cos(paramAngle) * self.radius, -sin(paramAngle) * self.radius)

    def getTangent(self, param):
        paramAngle = self.startAngle + param * 0.5 * pi
        return [sin(paramAngle), cos(paramAngle)] #??
        
    def draw(self):
        DrawLine(self.center, self.getPoint(0))
        count = 10
        for i in range(0, count):
            pointA = self.getPoint(i / count)
            pointB = self.getPoint((i+1) / count)
            DrawLine(pointA, pointB)



class TeardropTarget:
    def __init__(self, pos, angle, width, height, cornerRadius, drill, layer) -> None:
        self.pos = pos
        self.angle = angle
        self.width = width
        self.height = height
        self.cornerRadius = cornerRadius
        self.drill = drill
        self.layer = layer

    def basis(self):
        # Compute basis or rotated pad
        basis0 = [cos(-self.angle), sin(-self.angle)] # angle is stored in radians
        basis1 = [-sin(-self.angle), cos(-self.angle)]
        return basis0, basis1

    def generateSegments(self):
        basis0, basis1 = self.basis()
        sw = self.width - 2.0 * self.cornerRadius
        sh = self.height - 2.0 * self.cornerRadius
        halfWidthVec = wxPoint(basis0[0] * sw * 0.5, basis0[1] * sw * 0.5)
        halfHeightVec = wxPoint(basis1[0] * sh * 0.5, basis1[1] * sh * 0.5)
        radiusWidthVec = wxPoint(basis0[0] * self.cornerRadius, basis0[1] * self.cornerRadius)
        radiusHeightVec = wxPoint(basis1[0] * self.cornerRadius, basis1[1] * self.cornerRadius)
        
        # Generate the 4 segments
        segments = []
        if sw > 0:
            segments.append(OutlineSegment(self.pos - halfWidthVec - halfHeightVec - radiusHeightVec, self.pos + halfWidthVec - halfHeightVec - radiusHeightVec))
            segments.append(OutlineSegment(self.pos + halfWidthVec + halfHeightVec + radiusHeightVec, self.pos - halfWidthVec + halfHeightVec + radiusHeightVec))
        if sh > 0:
            segments.append(OutlineSegment(self.pos + halfWidthVec - halfHeightVec + radiusWidthVec, self.pos + halfWidthVec + halfHeightVec + radiusWidthVec))
            segments.append(OutlineSegment(self.pos - halfWidthVec + halfHeightVec - radiusWidthVec, self.pos - halfWidthVec - halfHeightVec - radiusWidthVec))
        return segments

    def generateArcs(self):
        arcs = []
        if self.cornerRadius > 0:
            basis0, basis1 = self.basis()
            sw = self.width - 2.0 * self.cornerRadius
            sh = self.height - 2.0 * self.cornerRadius
            # Generate the 4 arcs
            arcs.append(OutlineArc(self.pos + wxPoint(-basis0[0] * sw * 0.5 - basis1[0] * sh * 0.5, -basis0[1] * sw * 0.5 - basis1[1] * sh * 0.5), \
                self.cornerRadius, self.angle + 0.5 * pi))
            arcs.append(OutlineArc(self.pos + wxPoint( basis0[0] * sw * 0.5 - basis1[0] * sh * 0.5,  basis0[1] * sw * 0.5 - basis1[1] * sh * 0.5), \
                self.cornerRadius, self.angle))
            arcs.append(OutlineArc(self.pos + wxPoint( basis0[0] * sw * 0.5 + basis1[0] * sh * 0.5,  basis0[1] * sw * 0.5 + basis1[1] * sh * 0.5), \
                self.cornerRadius, self.angle - 0.5 * pi))
            arcs.append(OutlineArc(self.pos + wxPoint(-basis0[0] * sw * 0.5 + basis1[0] * sh * 0.5, -basis0[1] * sw * 0.5 + basis1[1] * sh * 0.5), \
                self.cornerRadius, self.angle + pi))
        return arcs

    def computeWidthsAlongLine(self, line : Line):
        basis0, basis1 = self.basis()
        sw = self.width - 2.0 * self.cornerRadius
        sh = self.height - 2.0 * self.cornerRadius
        # Generate the 4 corners
        corners = []
        for b0 in [-1, 1]:
            for b1 in [-1, 1]:
                corner = wxPoint(self.pos.x + basis0[0] * b0 * sw * 0.5 + basis1[0] * b1 * sh * 0.5, \
                    self.pos.y + basis0[1] * b0 * sw * 0.5 + basis1[1] * b1 * sh * 0.5)
                corners.append(corner)

        minDist = 0
        maxDist = 0
        lineNormal = [line.tangent[1], -line.tangent[0]] # 90 deg clockwise from tangent
        for corner in corners:
            # compute signed distance to line
            signedDist = DotProduct(corner - line.point, lineNormal)
            if signedDist + self.cornerRadius > maxDist:
                maxDist = signedDist + self.cornerRadius
            if signedDist - self.cornerRadius < minDist:
                minDist  = signedDist - self.cornerRadius

        return minDist, maxDist

    def computeMinMaxAnglesFromPoint(self, point : wxPoint):
        basis0, basis1 = self.basis()
        sw = self.width - 2.0 * self.cornerRadius
        sh = self.height - 2.0 * self.cornerRadius

        deltaToCenter = NormalizeVector(point - self.pos)
        normalVec = [-deltaToCenter[1], deltaToCenter[0]]

        minDot = 0
        minVec = deltaToCenter
        maxDot = 0
        maxVec = deltaToCenter

        # Generate the 4 corners
        for b0 in [-1, 1]:
            for b1 in [-1, 1]:
                corner = wxPoint(self.pos.x + basis0[0] * b0 * sw * 0.5 + basis1[0] * b1 * sh * 0.5, \
                    self.pos.y + basis0[1] * b0 * sw * 0.5 + basis1[1] * b1 * sh * 0.5)

                deltaToPoint = point - corner
                rightAngleVec = [-deltaToPoint[1], deltaToPoint[0]]
                d = sqrt(deltaToPoint[0]*deltaToPoint[0]+deltaToPoint[1]*deltaToPoint[1])
                if d >= self.cornerRadius :
                    rho = self.cornerRadius/d
                    ad = rho*rho
                    bd = rho*sqrt(1-ad)
                    T1 = [corner[0] + ad*deltaToPoint[0] + bd*rightAngleVec[0], corner[1] + ad*deltaToPoint[1] + bd*rightAngleVec[1]]
                    T2 = [corner[0] + ad*deltaToPoint[0] - bd*rightAngleVec[0], corner[1] + ad*deltaToPoint[1] - bd*rightAngleVec[1]]
                    vec1 = NormalizeVector(wxPoint(T1[0], T1[1]) - point)
                    vec2 = NormalizeVector(wxPoint(T2[0], T2[1]) - point)
                    dot1 = DotProduct(vec1, normalVec)
                    dot2 = DotProduct(vec2, normalVec)
                    if dot1 > maxDot:
                        maxDot = dot1
                        maxVec = vec1
                    if dot2 > maxDot:
                        maxDot = dot2
                        maxVec = vec2
                    if dot1 < minDot:
                        minDot = dot1
                        minVec = vec1
                    if dot2 < minDot:
                        minDot = dot2
                        minVec = vec2

        return (minVec, maxVec)
        

    def hitTest(self, point: wxPoint):
        # Project point to local reference frame
        basis0, basis1 = self.basis()
        hw = self.width * 0.5
        hh = self.height * 0.5
        hsw = hw - self.cornerRadius
        hsh = hh - self.cornerRadius
        delta = point - self.pos
        localPoint = [DotProduct(delta, basis0), DotProduct(delta, basis1)]
        if localPoint[0] < -hw or localPoint[0] > hw or localPoint[1] < -hh or localPoint[1] > hh:
            # point outside bounding rectangle
            return False
        elif localPoint[0] > -hw and localPoint[0] < hw and localPoint[1] > -hsh and localPoint[1] < hsh:
            # point inside inscribed rectangle
            return True
        elif localPoint[0] > -hsw and localPoint[0] < hsw and localPoint[1] > -hh and localPoint[1] < hh:
            # point inside inscribed rectangle
            return True
        else:
            # test against corners
            for b0 in [-1, 1]:
                for b1 in [-1, 1]:
                    corner = wxPoint(self.pos.x + basis0[0] * b0 * hsw + basis1[0] * b1 * hsh, \
                        self.pos.y + basis0[1] * b0 * hsw + basis1[1] * b1 * hsh)
                    if PointDistance(corner, point) < self.cornerRadius:
                        return True
            return False


    def computeIntersection(self, line: Line):

        # Iterate over the arcs and segments of the pad and find the closest intersection point to the line's anchor point
        shapes = self.generateArcs() + self.generateSegments()
        closestIntersectionShapeIndex = -1
        closestIntersectionDist = FromMM(1000) # should have float max or something accessible...
        closestIntersectionPoint = wxPoint(0,0)
        closestIntersectionTangent = wxPoint(0,0)

        for i in range(len(shapes)):
            shape = shapes[i]
            #shape.draw()
            res = shape.intersect(line)
            if res is not None:
                (pt, param) = res
                dist = PointDistance(pt, line.point)
                if dist < closestIntersectionDist:
                    closestIntersectionShapeIndex = i
                    closestIntersectionDist = dist
                    closestIntersectionPoint = pt
                    closestIntersectionTangent = shape.getTangent(param)

        if closestIntersectionShapeIndex != -1:
            return closestIntersectionPoint, closestIntersectionTangent
        else:
            return None


    def computePointsOnPadOutline(self, line: Line, vpercent):
        lineNormal = [line.tangent[1], -line.tangent[0]] # 90 deg clockwise from tangent
        maxDist, minDist = self.computeWidthsAlongLine(line)

        # Create two parallel lines above and below the track line passed in
        lineAbove = Line(wxPoint(line.point.x + lineNormal[0] * maxDist * vpercent, line.point.y + lineNormal[1] * maxDist * vpercent), line.tangent)
        lineBelow = Line(wxPoint(line.point.x + lineNormal[0] * minDist * vpercent, line.point.y + lineNormal[1] * minDist * vpercent), line.tangent)

        # Find intersection points between the lines and the pad outline
        resAbove = self.computeIntersection(lineAbove)
        resBelow = self.computeIntersection(lineBelow)
        return resAbove, resBelow


def __GetAllVias(board):
    """Just retreive all via from the given board"""
    vias = []
    vias_selected = []
    for item in board.GetTracks():
        if item.GetClass() == "PCB_VIA":
            pos = item.GetPosition()
            angle = 0
            width = item.GetWidth()
            height = width
            cornerRadius = width * 0.5
            drill = PCB_VIA(item).GetDrillValue()
            layer = -1
            target = TeardropTarget(pos, angle, width, height, cornerRadius, drill, layer)
            vias.append(target)
            if item.IsSelected():
                vias_selected.append(target)
    return vias, vias_selected


def __GetAllPads(board, filters=[]):
    """Just retreive all pads from the given board"""
    pads = []
    pads_selected = []
    for pad in board.GetPads():
        if pad.GetAttribute() in filters:
            pos = pad.GetPosition()
            angle = pad.GetOrientationRadians()
            width = pad.GetSizeX()
            height = pad.GetSizeY()
            drill = 0
            if pad.GetShape() == PAD_SHAPE_RECT:
                cornerRadius = 0
            elif pad.GetShape() == PAD_SHAPE_CIRCLE:
                cornerRadius = pad.GetSizeX() * 0.5
            else:
                cornerRadius = pad.GetRoundRectCornerRadius()
            # See where the pad is
            if pad.GetAttribute() == PAD_ATTRIB_SMD:
                # Cannot use GetLayer here because it returns the non-flipped
                # layer. Need to get the real layer from the layer set
                cu_stack = pad.GetLayerSet().CuStack()
                if len(cu_stack) == 0:
                    # The pad is not on a Copper layer
                    continue
                layer = cu_stack[0]
            else:
                layer = -1
            target = TeardropTarget(pos, angle, width, height, cornerRadius, drill, layer)
            pads.append(target)
            if pad.IsSelected():
                pads_selected.append(target)
    return pads, pads_selected


def __GetAllTeardrops(board):
    """Just retrieves all teardrops of the current board classified by net"""
    teardrops_zones = {}
    for zone in [board.GetArea(i) for i in range(board.GetAreaCount())]:
        if zone.GetPriority() == MAGIC_TEARDROP_ZONE_ID:
            netname = zone.GetNetname()
            if netname not in teardrops_zones.keys():
                teardrops_zones[netname] = []
            teardrops_zones[netname].append(zone)
    return teardrops_zones


def __DoesTeardropBelongTo(teardrop, track, via):
    """Return True if the teardrop covers given track AND via"""
    # First test if the via belongs to the teardrop
    if not teardrop.HitTest(via.pos):
        return False
    # In a second time, test if the track belongs to the teardrop
    if not track.HitTest(teardrop.GetBoundingBox().GetCenter()):
        return False
    return True


def __Zone(board, points, track):
    """Add a zone to the board"""
    z = ZONE(board)

    # Add zone properties
    z.SetLayer(track.GetLayer())
    z.SetNetCode(track.GetNetCode())
    z.SetLocalClearance(track.GetLocalClearance(track.GetClass()))
    z.SetMinThickness(25400)  # The minimum
    z.SetPadConnection(ZONE_CONNECTION_FULL)
    z.SetCornerSmoothingType(ZONE_SETTINGS.SMOOTHING_NONE)
    z.SetFillMode(ZONE_FILL_MODE_POLYGONS)
    z.SetIsFilled(True)
    z.SetPriority(MAGIC_TEARDROP_ZONE_ID)
    ol = z.Outline()
    ol.NewOutline()

    for p in points:
        ol.Append(p.x, p.y)

    return z


def __Bezier(p1, p2, p3, p4, n=20.0):
    n = float(n)
    pts = []
    for i in range(int(n)+1):
        t = i/n
        a = (1.0 - t)**3
        b = 3.0 * t * (1.0-t)**2
        c = 3.0 * t**2 * (1.0-t)
        d = t**3

        x = int(a * p1[0] + b * p2[0] + c * p3[0] + d * p4[0])
        y = int(a * p1[1] + b * p2[1] + c * p3[1] + d * p4[1])
        pts.append(wxPoint(x, y))
    return pts



def __ComputeCurved(vpercent, w, vec, via, pts, segs):
    """Compute the curves part points"""

    # A and B are points on the track
    # C and E are points on the via
    # D is midpoint behind the via centre

    radius = via.width/2
    minVpercent = float(w*2) / float(via.width)
    weaken = (vpercent/100.0 - minVpercent) / (1-minVpercent) / radius

    biasBC = 0.5 * PointDistance(pts[1], pts[2])
    biasAE = 0.5 * PointDistance(pts[4], pts[0])

    vecC = pts[2] - via.pos
    tangentC = [pts[2][0] - vecC[1]*biasBC*weaken,
                pts[2][1] + vecC[0]*biasBC*weaken]
    vecE = pts[4] - via.pos
    tangentE = [pts[4][0] + vecE[1]*biasAE*weaken,
                pts[4][1] - vecE[0]*biasAE*weaken]

    tangentB = [pts[1][0] - vec[0]*biasBC, pts[1][1] - vec[1]*biasBC]
    tangentA = [pts[0][0] - vec[0]*biasAE, pts[0][1] - vec[1]*biasAE]

    curve1 = __Bezier(pts[1], tangentB, tangentC, pts[2], n=segs)
    curve2 = __Bezier(pts[4], tangentE, tangentA, pts[0], n=segs)

    return curve1 + [pts[3]] + curve2

def __FindTouchingTrack(t1, endpoint, trackLookup):
    """Find a track connected to the end of another track"""
    match = 0
    matches = 0
    ret = False, None
    for t2 in trackLookup[t1.GetLayer()][t1.GetNetname()]:
        # The track object can change, this seems like the only
        # reliable way to test if tracks are the same
        if t2.GetStart() == t1.GetStart() and t2.GetEnd() == t1.GetEnd():
            continue
        match = t2.IsPointOnEnds(endpoint, 10)
        if match:
            # if faced with a Y junction, stop here
            matches += 1
            if matches > 1:
                return False, None
            ret = match, t2
    return ret

def __GetTrackStartTangent(track):
    start = track.GetStart()
    end = track.GetEnd()
    if type(track) == PCB_ARC:
        # startAngle, endAngle are the absolute start and end.
        # angle is the included angle of the arc, negative if anticlockwise
        angle      = track.GetAngle()
        startAngle = track.GetArcAngleStart()

        posAngle = startAngle

        # angle is returned in units of TenthsOfADegree
        posAngle *= pi/1800
        pcos = cos(posAngle)
        psin = sin(posAngle)
        # the vector points from start towards end
        # posAngle points from centre to pos, so rotate by 90 degrees
        if angle > 0:
            vec = [ -psin, pcos ]
        else:
            vec = [ psin, -pcos ]
    else:
        delta = end - start
        vec = NormalizeVector(delta)

    return vec

def __GetTrackEndTangent(track):
    start = track.GetStart()
    end = track.GetEnd()
    if type(track) == PCB_ARC:
        # startAngle, endAngle are the absolute start and end.
        # angle is the included angle of the arc, negative if anticlockwise
        angle      = track.GetAngle()
        endAngle = track.GetArcAngleEnd()

        posAngle = endAngle

        # angle is returned in units of TenthsOfADegree
        posAngle *= pi/1800
        pcos = cos(posAngle)
        psin = sin(posAngle)
        # the vector points from start towards end
        # posAngle points from centre to pos, so rotate by 90 degrees
        if angle > 0:
            vec = [ -psin, pcos ]
        else:
            vec = [ psin, -pcos ]
    else:
        delta = end - start
        vec = NormalizeVector(delta)

    return vec

def __FindPositionAndVectorAlongTrack(track, currentEndPoint, dist):
    """ return the x,y position and direction vector at a point on an arc """
    length = track.GetLength()
    start = track.GetStart()
    end = track.GetEnd()

    # Check if track is reversed
    distStart = PointDistance(currentEndPoint, start)
    distEnd = PointDistance(currentEndPoint, end)
    trackReversed = distEnd < distStart
    if trackReversed:
        # Swap some stuff since reversed
        start, end = end, start

    # Check that track is long enough
    if dist > length:
        return (False, end, None)

    percent = dist / length

    if type(track) == PCB_ARC:
        radius     = track.GetRadius()
        arcCenter  = track.GetPosition() # or maybe track.GetCenter()

        # startAngle, endAngle are the absolute start and end.
        # angle is the included angle of the arc, negative if anticlockwise
        angle      = track.GetAngle()
        startAngle = track.GetArcAngleStart()

        if trackReversed:
            angle = -angle
            startAngle = track.GetArcAngleEnd()

        posAngle = startAngle + angle * percent

        # angle is returned in units of TenthsOfADegree
        posAngle *= pi/1800
        pcos = cos(posAngle)
        psin = sin(posAngle)

        newX = arcCenter.x + pcos * radius
        newY = arcCenter.y + psin * radius

        # the vector points from start towards end
        # posAngle points from centre to pos, so rotate by 90 degrees
        if angle > 0:
            vec = [ -psin, pcos ]
        else:
            vec = [ psin, -pcos ]
    else:
        delta = end - start

        vec = NormalizeVector(delta)

        newX = start.x + delta.x * percent
        newY = start.y + delta.y * percent

    return (True, wxPoint(newX, newY), vec)


def __WalkTracks(startTrack, startPoint, dist, trackLookup):
    currentDist = dist
    currentTrack = startTrack
    currentPoint = startPoint
    loopCount = 0
    while True:
        (res, currentPoint, currentVec) = __FindPositionAndVectorAlongTrack(currentTrack, currentPoint, currentDist)
        if res:
            return (True, currentTrack, currentPoint, currentVec)
        else:
            currentDist -= currentTrack.GetLength()
            res, currentTrack = __FindTouchingTrack(currentTrack, currentPoint, trackLookup)
            if not res:
                return (False, currentTrack, currentPoint, None)
            # else iterate on the next track

        loopCount += 1
        if loopCount > 100:
            print("Infinite loop in __WalkTracks")
            return (False, currentTrack, currentPoint, None)



def __WalkTracksUntilOutOfPad(startTrack, startPoint, via, distIncr, maxDist, trackLookup):
    currentTrackDist = 0
    totalDist = 0
    currentTrack = startTrack
    currentPoint = startPoint
    currentVec = None
    loopCount = 0

    while via.hitTest(currentPoint):
        currentTrackDist += distIncr
        (res, currentPoint, currentVec) = __FindPositionAndVectorAlongTrack(currentTrack, currentPoint, currentTrackDist)
        if not res:
            
            loopCount += 1
            if loopCount > 100:
                print("Infinite loop in __WalkTracksUntilOutOfPad")
                return (False, currentTrack, currentPoint, maxDist, None)

            totalDist += currentTrack.GetLength()

            if totalDist > maxDist:
                print("Walked too far looking for edge of pad")
                return (False, currentTrack, currentPoint, maxDist, None)

            currentTrackDist = 0
            res, currentTrack = __FindTouchingTrack(currentTrack, currentPoint, trackLookup)
            if not res:
                return (False, currentTrack, currentPoint, totalDist, None)
            # else iterate on the next track

    return (True, currentTrack, currentPoint, totalDist + currentTrackDist, currentVec)

def __ComputePoints(track, via, hpercent, vpercent, segs, follow_tracks, trackLookup, noBulge):
    """Compute all teardrop points"""
    start = track.GetStart()
    end = track.GetEnd()
    minRadius = min(via.width/2.0, via.height/2.0)
    maxRadius = max(via.width/2.0, via.height/2.0)
    w = track.GetWidth()/2

    if vpercent > 100:
        vpercent = 100

    startVec = __GetTrackStartTangent(track)
    # Check if track is reversed
    distStart = PointDistance(via.pos, start)
    distEnd = PointDistance(via.pos, end)
    if distEnd < distStart:
        # Swap some stuff since reversed
        start, end = end, start
        startVec = __GetTrackEndTangent(track)

    # Find the first point along the track outside the via/pad
    (res, trk, ptOnOutline, distToOutline, vecOnOutline) = __WalkTracksUntilOutOfPad(track, start, via, FromMM(0.01), maxRadius * 2, trackLookup)
    if not res:
        print("Not adding teardrop because cannot get point outside pad/via")
        return False

    # Now we are out of the pad, find a point hpercent away
    (res, trk, ptOnTrack, vecOnTrack) = __WalkTracks(track, start, distToOutline + maxRadius, trackLookup)
    if not res:
        print("Not adding teardrop because cannot get point far enough from pad/via")
        return False

    (minVec, maxVec) = via.computeMinMaxAnglesFromPoint(ptOnTrack)
    minAngle = atan2(-minVec[1], minVec[0])
    maxAngle = atan2(-maxVec[1], maxVec[0])

    # handle vpercent
    totalAngle = abs(normalizeAngle(maxAngle - minAngle))
    angleAdjust = totalAngle * (1 - vpercent / 100) * 0.5

    minAngle += angleAdjust
    minDelta = [cos(minAngle), -sin(minAngle)]

    maxAngle -= angleAdjust
    maxDelta = [cos(maxAngle), -sin(maxAngle)]

    # find point on the track, sharp end of the teardrop
    pointB = ptOnTrack + wxPoint( vecOnTrack[1]*w, -vecOnTrack[0]*w)
    pointA = ptOnTrack + wxPoint(-vecOnTrack[1]*w,  vecOnTrack[0]*w)

    # via side points
    minLine = Line(ptOnTrack, minDelta)
    resC = via.computeIntersection(minLine)
    maxLine = Line(ptOnTrack, maxDelta)
    resE = via.computeIntersection(maxLine)
    if resC == None or resE == None:
        print("Not adding teardrop because no intersection")
        return False

    pointC = resC[0]
    tanC = resC[1]
    biasBC = 0.5 * PointDistance(pointB, pointC)
    tangentPointC = [pointC[0] + tanC[0] * biasBC, pointC[1] + tanC[1] * biasBC]
    tangentPointB = [pointB[0] - vecOnTrack[0] * biasBC, pointB[1] - vecOnTrack[1] * biasBC]

    pointE = resE[0]
    tanE = resE[1]
    biasAE = 0.5 * PointDistance(pointE, pointA)
    tangentPointE = [pointE[0] - tanE[0] * biasAE, pointE[1] - tanE[1] * biasAE]
    tangentPointA = [pointA[0] - vecOnTrack[0] * biasAE, pointA[1] - vecOnTrack[1] * biasAE]

    # Introduce a last point in order to cover the via centre.
    # If not, the zone won't be filled
    pointD = wxPoint(via.pos.x - startVec[0] * FromMM(0.1), via.pos.y - startVec[1] * FromMM(0.1))

    if segs > 2:
        # A and B are points on the track
        # C and E are points on the via
        # D is midpoint behind the via centre
        curve1 = __Bezier(pointB, tangentPointB, tangentPointC, pointC, n=segs)
        curve2 = __Bezier(pointE, tangentPointE, tangentPointA, pointA, n=segs)

        pts = curve1 + [pointD] + curve2

    return pts


def __IsViaAndTrackInSameNetZone(pcb, via, track):
    """Return True if the given via + track is located inside a zone of the
    same netname"""
    for zone in [pcb.GetArea(i) for i in range(pcb.GetAreaCount())]:
        # Exclude other Teardrops to speed up the process
        if zone.GetPriority() == MAGIC_TEARDROP_ZONE_ID:
            continue

        # Only consider zones on the same layer
        if not zone.IsOnLayer(track.GetLayer()):
            continue

        if (zone.GetNetname() == track.GetNetname()):
            if zone.Outline().Contains(VECTOR2I(*via.pos)):
                return True
    return False


def RebuildAllZones(pcb):
    """Rebuilt all zones"""
    filler = ZONE_FILLER(pcb)
    filler.Fill(pcb.Zones())


def SetTeardrops(hpercent=50, vpercent=95, segs=10, pcb=None, use_smd=False,
                 discard_in_same_zone=True, follow_tracks=True, noBulge=True):
    """Set teardrops on a teardrop free board"""
    if pcb is None:
        pcb = GetBoard()

    pad_types = [PAD_ATTRIB_PTH] + [PAD_ATTRIB_SMD]*use_smd
    vias = __GetAllVias(pcb)[0] + __GetAllPads(pcb, pad_types)[0]
    vias_selected = __GetAllVias(pcb)[1] + __GetAllPads(pcb, pad_types)[1]
    if len(vias_selected) > 0:
        vias = vias_selected

    trackLookup = {}
    if follow_tracks:
        for t in pcb.GetTracks():
            if isinstance(t, PCB_TRACK):
                net = t.GetNetname()
                layer = t.GetLayer()

                if layer not in trackLookup:
                    trackLookup[layer] = {}
                if net not in trackLookup[layer]:
                    trackLookup[layer][net] = []
                trackLookup[layer][net].append(t)

    teardrops = __GetAllTeardrops(pcb)
    count = 0

    for track in [t for t in pcb.GetTracks() if isinstance(t, PCB_TRACK)]:
        for via in [v for v in vias if track.IsPointOnEnds(v.pos, int(v.width/2))]:
            if track.GetWidth() >= via.width * vpercent / 100:
                continue

            if track.IsPointOnEnds(via.pos, int(via.width/2)) == \
               STARTPOINT | ENDPOINT:
                # both start and end are within the via
                continue

            found = False
            if track.GetNetname() in teardrops.keys():
                for teardrop in teardrops[track.GetNetname()]:
                    if __DoesTeardropBelongTo(teardrop, track, via):
                        found = True
                        break

            # Discard case where pad and track are on different layers, or the
            # pad have no copper at all (paste pads).
            if (via.layer != -1) and (via.layer != track.GetLayer()):
                continue

            # Discard case where pad/via is within a zone with the same netname
            # WARNING: this can severely reduce performance
            if discard_in_same_zone and \
               __IsViaAndTrackInSameNetZone(pcb, via, track):
                continue

            if not found:
                print("Computing points for " + str(via) + " and " + str(track))
                coor = __ComputePoints(track, via, hpercent, vpercent, segs,
                                       follow_tracks, trackLookup, noBulge)
                if coor:
                    pcb.Add(__Zone(pcb, coor, track))
                    count += 1

    RebuildAllZones(pcb)
    return count


def RmTeardrops(pcb=None):
    """Remove all teardrops"""

    if pcb is None:
        pcb = GetBoard()

    count = 0
    teardrops = __GetAllTeardrops(pcb)
    for netname in teardrops:
        for teardrop in teardrops[netname]:
            pcb.Remove(teardrop)
            count += 1

    RebuildAllZones(pcb)
    return count
