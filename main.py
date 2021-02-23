import numpy as np
import struct

# parametric curves

def trefoil_knot(t):
    t = t * 2 * np.pi
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    z = -np.sin(3 * t)
    return np.stack([x, y, z], axis=-1)

def lissajous_knot(fx, fy, fz, px, py, pz):
    def position(t):
        t = t * 2 * np.pi
        x = np.cos(fx * t + px * 2 * np.pi)
        y = np.cos(fy * t + py * 2 * np.pi)
        z = np.cos(fz * t + pz * 2 * np.pi)
        return np.stack([x, y, z], axis=-1)
    return position

def spiral(r, k):
    def position(t):
        t = t * 2 * np.pi
        x = np.cos(k * t) * r
        y = np.sin(k * t) * r
        z = t
        return np.stack([x, y, z], axis=-1)
    return position

# cross-section profiles

def elliptical_profile(rx, ry, n):
    a = np.linspace(0, 2 * np.pi, n, False)
    x = np.cos(a) * rx
    y = np.sin(a) * ry
    return np.stack([x, y], axis=-1)

def rectangular_profile(rx, ry):
    return np.array([(-rx, -ry), (rx, -ry), (rx, ry), (-rx, ry)])

# mesh generation

def mesh(curve, profile, n):
    eps = 1e-9
    def normalize(v):
        return v / np.linalg.norm(v, axis=-1).reshape((-1, 1))
    def cap(p, section):
        p0 = section
        p1 = np.roll(section, -1, axis=0)
        a = np.zeros((len(section) * 3, 3))
        a[0::3,:], a[1::3,:], a[2::3,:] = p, p0, p1
        return a
    closed = np.linalg.norm(curve(0) - curve(1)) < eps
    t = np.linspace(0, 1, n + 1)
    p = curve(t)
    d = normalize(curve(t + eps) - curve(t - eps))
    up = normalize(p)
    u = normalize(np.cross(up, d))
    v = normalize(np.cross(d, u))
    p = p.reshape((-1, 1, 3))
    u = u.reshape((-1, 1, 3))
    v = v.reshape((-1, 1, 3))
    px = profile[:,0].reshape((1, -1, 1))
    py = profile[:,1].reshape((1, -1, 1))
    sections = p + u * px + v * py
    i, j, _ = sections.shape
    p00 = sections.reshape((-1, 3))
    p01 = np.roll(sections, -1, axis=0).reshape((-1, 3))
    p10 = np.roll(sections, -1, axis=1).reshape((-1, 3))
    p11 = np.roll(sections, (-1, -1), axis=(0, 1)).reshape((-1, 3))
    a = np.zeros((i * j * 3 * 2, 3))
    a[0::6,:], a[1::6,:], a[2::6,:] = p00, p10, p11
    a[3::6,:], a[4::6,:], a[5::6,:] = p00, p11, p01
    arrs = [a[:-j*6]]
    if not closed:
        arrs.append(cap(p[0], np.flip(sections[0], axis=0)))
        arrs.append(cap(p[-1], sections[-1]))
    return np.concatenate(arrs)

def write_binary_stl(path, points):
    n = len(points) // 3
    points = np.array(points, dtype='float32').reshape((-1, 3, 3))
    normals = np.cross(points[:,1] - points[:,0], points[:,2] - points[:,0])
    normals /= np.linalg.norm(normals, axis=1).reshape((-1, 1))
    dtype = np.dtype([
        ('normal', ('<f', 3)),
        ('points', ('<f', (3, 3))),
        ('attr', '<H'),
    ])
    a = np.zeros(n, dtype=dtype)
    a['points'] = points
    a['normal'] = normals
    with open(path, 'wb') as fp:
        fp.write(b'\x00' * 80)
        fp.write(struct.pack('<I', n))
        fp.write(a.tobytes())

# main!

def main():
    curve = trefoil_knot
    # curve = spiral(3, 3)
    # curve = lissajous_knot(3, 4, 5, 0.55, 0.15, 0)
    profile = elliptical_profile(0.125, 0.125, 360)
    # profile = rectangular_profile(0.125, 0.125)
    points = mesh(curve, profile, 1024)
    write_binary_stl('out.stl', points)

if __name__ == '__main__':
    main()
