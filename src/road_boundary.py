"""Read-only sampled vehicle-footprint coverage by SUMO road/junction surfaces.

No collision or reward intervention. Subtract convex road tiles from the body;
normal lane-divider crossings are covered by the union of all lanes.
"""
import math
import xml.etree.ElementTree as ET
from functools import lru_cache
from traci.exceptions import TraCIException


def area(poly):
    return abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(poly,poly[1:]+poly[:1])))/2 if len(poly)>2 else 0.0


def cross(a,b,p):
    return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])


def ccw(poly):
    signed=sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(poly,poly[1:]+poly[:1]))
    return poly if signed>=0 else list(reversed(poly))


def clip(poly,a,b,inside):
    result=[]
    for p,q in zip(poly,poly[1:]+poly[:1]):
        cp,cq=cross(a,b,p),cross(a,b,q)
        ip,iq=(cp>=0,cq>=0) if inside else (cp<=0,cq<=0)
        if ip:result.append(p)
        if ip != iq:
            t=cp/(cp-cq)
            result.append((p[0]+t*(q[0]-p[0]),p[1]+t*(q[1]-p[1])))
    return result


def subtract(poly,tile):
    outside=[]; remainder=poly
    for a,b in zip(tile,tile[1:]+tile[:1]):
        fragment=clip(remainder,a,b,False)
        if area(fragment)>1e-10:outside.append(fragment)
        remainder=clip(remainder,a,b,True)
        if not remainder:break
    return outside


def triangulate(poly):
    p=ccw(poly); result=[]
    while len(p)>3:
        found=False
        for i,b in enumerate(p):
            a,c=p[i-1],p[(i+1)%len(p)]
            if cross(a,b,c)<=1e-10:continue
            others=[v for j,v in enumerate(p) if j not in ((i-1)%len(p),i,(i+1)%len(p))]
            if any(cross(a,b,v)>=0 and cross(b,c,v)>=0 and cross(c,a,v)>=0 for v in others):continue
            result.append([a,b,c]);p.pop(i);found=True;break
        if not found:raise ValueError('Unsupported junction polygon')
    if area(p)>1e-10:result.append(p)
    return result


def bbox(p):return min(x for x,y in p),min(y for x,y in p),max(x for x,y in p),max(y for x,y in p)


def overlaps(a,b):return not (a[2]<b[0] or b[2]<a[0] or a[3]<b[1] or b[3]<a[1])


@lru_cache(maxsize=4)
def road_tiles(path):
    root=ET.parse(path).getroot();tiles=[]
    for edge in root.findall('edge'):
        if edge.get('function')=='internal':continue  # junction polygon is authoritative
        for lane in edge.findall('lane'):
            allowed=lane.get('allow','').split();disallowed=lane.get('disallow','').split()
            if (allowed and 'passenger' not in allowed and 'all' not in allowed) or 'passenger' in disallowed:continue
            pts=[tuple(map(float,v.split(','))) for v in lane.get('shape','').split()]
            w=float(lane.get('width','3.2'))/2
            normals=[]
            for a,b in zip(pts,pts[1:]):
                dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy)
                if length==0:raise ValueError('Zero length lane shape segment')
                normals.append((-dy/length,dx/length))
            sides=[]
            for i,p in enumerate(pts):
                n1=normals[max(0,i-1)];n2=normals[min(i,len(normals)-1)]
                nx,ny=n1[0]+n2[0],n1[1]+n2[1];den=nx*n2[0]+ny*n2[1]
                if den<=1e-9:raise ValueError('Unsupported lane shape reversal')
                ox,oy=w*nx/den,w*ny/den
                sides.append(((p[0]+ox,p[1]+oy),(p[0]-ox,p[1]-oy)))
            for a,b in zip(sides,sides[1:]):tiles.append(ccw([a[0],a[1],b[1],b[0]]))
    for junction in root.findall('junction'):
        shape=[tuple(map(float,v.split(','))) for v in junction.get('shape','').split()]
        if len(shape)>=3:tiles.extend(triangulate(shape))
    return tuple((tile,bbox(tile)) for tile in tiles)


def footprint(position,angle,length,width):
    if (any(not math.isfinite(v) for v in (*position,angle,length,width))
            or min(length,width) <= 0):
        raise ValueError('Invalid vehicle footprint data')
    x,y=position;r=math.radians(angle);fx,fy=math.sin(r),math.cos(r);lx,ly=-fy,fx
    return ccw([(x+sx*lx*width/2-back*fx*length,y+sx*ly*width/2-back*fy*length)
                for sx,back in [(1,0),(-1,0),(-1,1),(1,1)]])


def coverage(body,tiles):
    bb=bbox(body);fragments=[body]
    for tile,tb in tiles:
        if not overlaps(bb,tb):continue
        fragments=[piece for fragment in fragments for piece in subtract(fragment,tile)]
        if not fragments:break
    uncovered=sum(area(p) for p in fragments)
    return {'status':'inside' if uncovered<=1e-4 else 'outside',
            'uncovered_area_m2':uncovered,'body_area_m2':area(body)}


def observe(base):
    if not base.ego_on_road:return {'status':'unknown','reason':'vehicle_not_available'}
    v=base.traci.vehicle;e=base.ego_id
    try:
        body=footprint(v.getPosition(e),v.getAngle(e),v.getLength(e),v.getWidth(e))
        result=coverage(body,road_tiles(str(base.network_path)))
        if (result['status'] == 'outside' and result['uncovered_area_m2'] <= 0.02
                and (v.getRoadID(e).startswith(':') or v.getLanePosition(e) <= v.getLength(e) + 0.05)):
            result.update(status='unknown', reason='small_junction_geometry_seam')
        result.update(road=v.getRoadID(e),lane=v.getLaneID(e),
                      lateral_position_m=v.getLateralLanePosition(e),
                      lateral_speed_mps=v.getLateralSpeed(e))
        return result
    except (ValueError,IndexError,ZeroDivisionError,OSError,ET.ParseError,TraCIException) as error:
        return {'status':'unknown','reason':str(error)}
