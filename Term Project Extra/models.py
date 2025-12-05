import math
from datetime import datetime

def parse_time(s):
    try:
        return datetime.fromisoformat(s.replace("Z",""))
    except:
        return None

def angular_distance_deg(ra1, dec1, ra2, dec2):
    ra1r = math.radians(ra1)
    dec1r = math.radians(dec1)
    ra2r = math.radians(ra2)
    dec2r = math.radians(dec2)

    cosd = (
        math.sin(dec1r) * math.sin(dec2r) +
        math.cos(dec1r) * math.cos(dec2r) * math.cos(abs(ra1r - ra2r))
    )
    cosd = max(-1, min(1, cosd))
    return math.degrees(math.acos(cosd))

class Observation:
    def __init__(self, oid, start, end, duration, ra, dec, weight):
        self.id = oid
        self.start = start
        self.end = end
        self.duration = duration
        self.ra = ra
        self.dec = dec
        self.weight = weight

    def __repr__(self):
        return f"Obs({self.id}, w={self.weight}, {self.start}->{self.end})"