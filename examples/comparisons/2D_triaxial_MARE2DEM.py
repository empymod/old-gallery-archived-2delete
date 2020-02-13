"""
MARE2DEM: 2D with tri-axial anisotropy
======================================

``MARE2DEM`` is an open-source, finite element 2.5D code for controlled-source
electromagnetic (CSEM) and magnetotelluric (MT) data, see `mare2dem.ucsd.edu
<https://mare2dem.ucsd.edu>`_. The ``MARE2DEM`` input- and output-files are
located in the data-directory.

"""
import emg3d
import discretize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.style.use('ggplot')
# sphinx_gallery_thumbnail_path = '_static/thumbs/MARE2DEM.png'

###############################################################################
# emg3d
# -----

src = [50, 0, -1950, 0, 0]  # Source location [x, y, z, azimuth, dip]
freq = 0.5                  # Frequency (Hz)


###############################################################################

# Create stretched grid
pgrid = discretize.TensorMesh(
    [[(100, 16, -1.08), (100, 100), (100, 12, 1.08)],
     [(50, 31, -1.08), (50, 2), (50, 31, 1.08)],
     [(100, 80), (100, 48, 1.03)]],
    x0=(-3275.0225685, 'C', -7000))

pgrid


###############################################################################

xx = (pgrid.gridCC[:, 0] > 0)*(pgrid.gridCC[:, 0] <= 6000)
zz = (pgrid.gridCC[:, 2] > -4200)*(pgrid.gridCC[:, 2] < -4000)


###############################################################################

# Background
res_x_full = 2*np.ones(pgrid.nC)
res_y_full = 1*np.ones(pgrid.nC)
res_z_full = 3*np.ones(pgrid.nC)

# Water - isotropic
res_x_full[pgrid.gridCC[:, 2] >= -2000] = 0.3
res_y_full[pgrid.gridCC[:, 2] >= -2000] = 0.3
res_z_full[pgrid.gridCC[:, 2] >= -2000] = 0.3

# Air - isotropic
res_x_full[pgrid.gridCC[:, 2] >= 0] = 1e12
res_y_full[pgrid.gridCC[:, 2] >= 0] = 1e12
res_z_full[pgrid.gridCC[:, 2] >= 0] = 1e12

# Target
res_x_full_tg = res_x_full.copy()
res_y_full_tg = res_y_full.copy()
res_z_full_tg = res_z_full.copy()
res_x_full_tg[xx*zz] = 200
res_y_full_tg[xx*zz] = 100
res_z_full_tg[xx*zz] = 300

pmodel = emg3d.utils.Model(pgrid, res_x_full, res_y_full, res_z_full)
pmodel_tg = emg3d.utils.Model(
        pgrid, res_x_full_tg, res_y_full_tg, res_z_full_tg)

pgrid.plot_3d_slicer(
        pmodel_tg.res_x, clim=[0.3, 300], zlim=[-6000, 500],
        pcolorOpts={'norm': LogNorm()})


###############################################################################
# Model background
# ````````````````

sfield = emg3d.utils.get_source_field(pgrid, src, freq, 0)
pfield = emg3d.solve(
        pgrid, pmodel, sfield, verb=3,
        semicoarsening=True, linerelaxation=True)


###############################################################################
# Model target
# ````````````

sfield_tg = emg3d.utils.get_source_field(pgrid, src, freq, 0)
pfield_tg = emg3d.solve(
        pgrid, pmodel_tg, sfield_tg, verb=3,
        semicoarsening=True, linerelaxation=True)


###############################################################################
# Load MARE2DEM result
# --------------------

dat = np.loadtxt('./data/triaxial.0.resp', skiprows=93, usecols=6)
mare = dat[::2] + 1j*dat[1::2]

bgdat = np.loadtxt('./data/triaxial-BG.0.resp', skiprows=93, usecols=6)
bgmare = bgdat[::2] + 1j*bgdat[1::2]

x = np.arange(80)/10+2.05

# Get corresponding emg3d offsets and responses
xx = pgrid.vectorCCx[36:-12]/1e3
if not np.allclose(x, xx):
    print("\n\n\n ===== ¡ Watch out, offsets are not the same ! ===== \n\n\n")

em3_bg = pfield.fx[36:-12, 32, 50]
em3_tg = pfield_tg.fx[36:-12, 32, 50]


###############################################################################
# Differences
# ```````````
#
# - In ``emg3d``, the source is a cell of 100x50x100 meters, with center at
#   (50, 0, -1950); center is same as ``MARE2DEM`` source location.
# - In ``MARE2DEM`` the receivers are at -1999.9 m depth, 10 m above the
#   sea-surface. In ``emg3d``, we take the edges with are at -2000 m, hence the
#   seafloor itself; the edges are 100 m long (but the response is normalized).

plt.figure(figsize=(9, 8))

plt.subplot(221)
plt.title(r'|Real response|')

plt.plot(x, np.abs(mare.real), '-', label='MARE2DEM target')
plt.plot(x, np.abs(bgmare.real), '-', label='MARE2DEM BG')

plt.plot(xx, np.abs(em3_tg.real), 'C4--', label='emg3d target')
plt.plot(xx, np.abs(em3_bg.real), 'C5--', label='emg3d BG')

# plt.yscale('symlog', linthreshy=5e-16, linscaley=0.5)
plt.yscale('log')
plt.ylabel('Amplitude (V/m)')
plt.xlabel('Offset (km)')
plt.legend()

plt.subplot(222)
plt.title(r'Relative error')

plt.semilogy(x, 100*np.abs((mare.real-em3_tg.real)/mare.real),
             '.-', label='target')
plt.semilogy(x, 100*np.abs((bgmare.real-em3_bg.real)/bgmare.real),
             '.-', label='background')

plt.ylabel('Rel. Error (%)')
plt.xlabel('Offset (km)')
plt.legend()

plt.subplot(223)
plt.title(r'|Imaginary response|')

plt.plot(x, np.abs(mare.imag), '-', label='MARE2DEM target')
plt.plot(x, np.abs(bgmare.imag), '-', label='MARE2DEM BG')

plt.plot(xx, np.abs(em3_tg.imag), 'C4--', label='emg3d target')
plt.plot(xx, np.abs(em3_bg.imag), 'C5--', label='emg3d BG')

# plt.yscale('symlog', linthreshy=5e-16, linscaley=0.5)
plt.yscale('log')
plt.ylabel('Amplitude (V/m)')
plt.xlabel('Offset (km)')
plt.legend()

plt.subplot(224)
plt.title(r'Relative error')

plt.semilogy(x, 100*np.abs((mare.imag-em3_tg.imag)/mare.imag),
             '.-', label='target')
plt.semilogy(x, 100*np.abs((bgmare.imag-em3_bg.imag)/bgmare.imag),
             '.-', label='background')

plt.ylabel('Rel. Error (%)')
plt.xlabel('Offset (km)')
plt.legend()

plt.tight_layout()
plt.show()


###############################################################################

emg3d.Report()
