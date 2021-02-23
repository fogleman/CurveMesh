import numpy as np
import struct

def trefoil_knot(t):
    t = t * 2 * np.pi
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    z = -np.sin(3 * t)
    return np.stack([x, y, z], axis=-1)

def elliptical_profile(rx, ry, n):
    a = np.linspace(0, 2 * np.pi, n, False)
    x = np.cos(a) * rx
    y = np.sin(a) * ry
    return np.stack([x, y], axis=-1)

def rectangular_profile(rx, ry):
    return np.array([(-rx, -ry), (rx, -ry), (rx, ry), (-rx, ry)])

def mesh(curve, profile, n):
    normalize = lambda v: v / np.linalg.norm(v, axis=-1).reshape((-1, 1))
    t = np.linspace(0, 1, n, False)
    p = curve(t)
    d = normalize(curve(t + 1e-9) - curve(t - 1e-9))
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
    p00 = sections
    p01 = np.roll(sections, 1, axis=0)
    p10 = np.roll(sections, 1, axis=1)
    p11 = np.roll(p01, 1, axis=1)
    a = np.zeros((i * j * 3 * 2, 3))
    a[0::6,:] = p00.reshape((-1, 3))
    a[1::6,:] = p10.reshape((-1, 3))
    a[2::6,:] = p11.reshape((-1, 3))
    a[3::6,:] = p00.reshape((-1, 3))
    a[4::6,:] = p11.reshape((-1, 3))
    a[5::6,:] = p01.reshape((-1, 3))
    return a

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

def main():
    curve = trefoil_knot
    profile = elliptical_profile(0.25, 0.5, 360)
    # profile = rectangular_profile(0.25, 0.5)
    points = mesh(curve, profile, 512)
    write_binary_stl('out.stl', points)

if __name__ == '__main__':
    main()
