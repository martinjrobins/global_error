
class CubicHermiteInterpolate:
    def __init__(self, t0, t1, y0, y1, dy0, dy1):
        self.h = t1-t0
        self.t0 = t0
        self.p0 = y0
        self.p1 = y1
        self.m0 = dy0
        self.m1 = dy1

    def __call__(self, raw_t):
        t = (raw_t - self.t0) / self.h
        t2 = t**2
        tmp = (1 - t)**2
        h00 = (1 + 2*t) * tmp
        h10 = t * tmp
        h01 = t2 * (3 - 2*t)
        h11 = t2 * (t - 1)
        return h00 * self.p0 \
            + h10 * self.h * self.m0 \
            + h01 * self.p1 \
            + h11 * self.h * self.m1

    def grad(self, raw_t):
        t = (raw_t - self.t0) / self.h
        t2 = t**2
        h00 = 6 * t2 - 6 * t
        h10 = 3 * t2 - 4 * t + 1
        h01 = -h00
        h11 = 3 * t2 - 2 * t
        return (
            h00 * self.p0
            + h10 * self.h * self.m0
            + h01 * self.p1
            + h11 * self.h * self.m1
        ) / self.h
