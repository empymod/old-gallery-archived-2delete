r"""
2. Magnetic source using an el. loop
====================================

Computing the :math:`E` and :math:`H` fields generated by a magnetic source

We know that we can get the magnetic fields from the electric fields using
Faraday's law, see :ref:`sphx_glr_gallery_magnetics_magnetic_field.py`.

However, what about computing the fields generated by a magnetic source?
There are two ways we can achieve that:

- **creating an electric loop source**, which is what we do in this example,
  or
- using the duality principle, see
  :ref:`sphx_glr_gallery_magnetics_magnetic_source_duality.py`.

We create a horizontal, electric loop source to generate a vertical magnetic
field in a homogeneous VTI fullspace, and compare it to the semi-analytical
solution of ``empymod``. (The code ``empymod`` is an open-source code which can
model CSEM responses for a layered medium including VTI electrical anisotropy,
see `empymod.github.io <https://empymod.github.io>`_.)

The method used here can be applied to an arbitrarily rotated, arbitrarily
shaped source loop.

"""
import emg3d
import empymod
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as sint
from matplotlib.colors import SymLogNorm
plt.style.use('ggplot')
# sphinx_gallery_thumbnail_path = '_static/thumbs/el-loop.png'


###############################################################################
def plot_data_rel(ax, name, data, x, vmin=-15., vmax=-7., mode="log"):
    """Plot function."""

    ax.set_title(name)
    ax.set_xlim(min(x)/1000, max(x)/1000)
    ax.set_ylim(min(x)/1000, max(x)/1000)
    ax.axis("equal")

    if isinstance(mode, str):
        if mode == "abs":
            cf = ax.pcolormesh(
                    x/1000, x/1000, np.log10(np.abs(data)), linewidth=0,
                    rasterized=True, cmap="viridis", vmin=vmin, vmax=vmax)
        else:
            cf = ax.pcolormesh(
                    x/1000, x/1000, data, linewidth=0, rasterized=True,
                    cmap="PuOr_r",
                    norm=SymLogNorm(linthresh=10**vmin,
                                    vmin=-10**vmax, vmax=10**vmax))
    else:
        cf = ax.pcolormesh(
                x/1000, x/1000, np.log10(data), vmin=vmin, vmax=vmax,
                linewidth=0, rasterized=True,
                cmap=plt.cm.get_cmap("RdBu_r", 8))

    return cf


###############################################################################
def plot_result_rel(depm, de3d, x, title, vmin=-15., vmax=-7., mode="log"):
    fig, axs = plt.subplots(figsize=(18, 10), nrows=2, ncols=3)

    if mode == "log":
        case = ""
    else:
        case = "|"

    # Plot Re(data)
    cf0 = plot_data_rel(axs[0, 0], r"(a) "+case+"Re(empymod)"+case,
                        depm.real, x, vmin, vmax, mode)
    plot_data_rel(axs[0, 1], r"(b) "+case+"Re(emg3d)"+case,
                  de3d.real, x, vmin, vmax, mode)
    cf2 = plot_data_rel(axs[0, 2], r"(c) Error real part",
                        np.abs((depm.real-de3d.real)/depm.real)*100, x,
                        vmin=-2, vmax=2, mode=True)

    # Plot Im(data)
    plot_data_rel(axs[1, 0], r"(d) "+case+"Im(empymod)"+case,
                  depm.imag, x, vmin, vmax, mode)
    plot_data_rel(axs[1, 1], r"(e) "+case+"Im(emg3d)"+case,
                  de3d.imag, x, vmin, vmax, mode)
    plot_data_rel(axs[1, 2], r"(f) Error imaginary part",
                  np.abs((depm.imag-de3d.imag)/depm.imag)*100,
                  x, vmin=-2, vmax=2, mode=True)

    # Colorbars
    fig.colorbar(cf0, ax=axs[0, :], label=r"$\log_{10}$ Amplitude (A/m)")
    cbar = fig.colorbar(cf2, ax=axs[1, :], label=r"Relative Error")
    cbar.set_ticks([-2, -1, 0, 1, 2])
    cbar.ax.set_yticklabels([r"$0.01\,\%$", r"$0.1\,\%$", r"$1\,\%$",
                             r"$10\,\%$", r"$100\,\%$"])

    # Axis label
    fig.text(0.4, 0.05, "Inline Offset (km)", fontsize=14)
    fig.text(0.08, 0.6, "Crossline Offset (km)", rotation=90, fontsize=14)

    # Title
    fig.suptitle(title, y=1, fontsize=20)
    plt.show()


###############################################################################
def plot_lineplot_ex(x, y, data, epm_fs, grid):
    xi = x.size//2
    yi = y.size//2

    fn = sint.interp1d(x, data[:, xi], bounds_error=False)
    # x1 = fn(grid.vectorNx)

    fn = sint.interp1d(y, data[yi, :], bounds_error=False)
    y1 = fn(grid.vectorNx)

    plt.figure(figsize=(15, 8))

    plt.plot(y/1e3, np.abs(epm_fs[yi, :]), 'C1', lw=3, label='Inline empymod')
    plt.plot(y/1e3, np.abs(data[yi, :]), 'k:', label='Inline emg3d')
    plt.plot(grid.vectorNx/1e3, np.abs(y1), 'k*', label='Grid points emg3d')

    plt.yscale('log')
    plt.title(r'Inline $H_x$', fontsize=20)
    plt.xlabel('Offset (km)', fontsize=14)
    plt.ylabel(r'|Amplitude (A/m)|', fontsize=14)
    plt.legend()
    plt.show()


###############################################################################
# Full-space model for a loop dipole
# ----------------------------------
#
# empymod
# ```````

# Survey parameters
x = (np.arange(1025))*5-2560
rx = np.repeat([x, ], np.size(x), axis=0)
ry = rx.transpose()

# Model parameters
resh = 1.              # Horizontal resistivity
aniso = np.sqrt(2.)    # Anisotropy
resv = resh*aniso**2   # Vertical resistivity
src = [0, 0, -300, 0, -90]  # Source: [x, y, z, azimuth, dip]
zrec = -400.           # Receiver depth
freq = 0.77            # Frequency
strength = np.pi       # Source strength

# Input for empymod
model = {
    'src': src,
    'depth': [],
    'res': resh,
    'aniso': aniso,
    'strength': strength,
    'freqtime': freq,
    'htarg': {'pts_per_dec': -1},
}

###############################################################################

rxx = rx.ravel()
ryy = ry.ravel()

# e-field
epm_fs_ex = -empymod.loop(rec=[rxx, ryy, zrec, 0, 0], mrec=False, verb=3,
                          **model).reshape(np.shape(rx))
epm_fs_ey = -empymod.loop(rec=[rxx, ryy, zrec, 90, 0], mrec=False, verb=1,
                          **model).reshape(np.shape(rx))
epm_fs_ez = -empymod.loop(rec=[rxx, ryy, zrec, 0, 90], mrec=False, verb=1,
                          **model).reshape(np.shape(rx))

# h-field
epm_fs_hx = empymod.loop(rec=[rxx, ryy, zrec, 0, 0], verb=1,
                         **model).reshape(np.shape(rx))
epm_fs_hy = empymod.loop(rec=[rxx, ryy, zrec, 90, 0], verb=1,
                         **model).reshape(np.shape(rx))
epm_fs_hz = empymod.loop(rec=[rxx, ryy, zrec, 0, 90], verb=1,
                         **model).reshape(np.shape(rx))


###############################################################################
# emg3d
# `````

# Get computation domain as a function of frequency (resp., skin depth)
hx_min, xdomain = emg3d.meshes.get_domain(x0=src[0], freq=0.1, min_width=20)
hz_min, zdomain = emg3d.meshes.get_domain(x0=src[2], freq=0.1, min_width=20)

# Create stretched grid
nx = 2**7
hx = emg3d.meshes.get_stretched_h(hx_min, xdomain, nx, src[0])
hy = emg3d.meshes.get_stretched_h(hx_min, xdomain, nx, src[1])
hz = emg3d.meshes.get_stretched_h(hz_min, zdomain, nx, src[2])
pgrid = emg3d.TensorMesh([hx, hy, hz], x0=(xdomain[0], xdomain[0], zdomain[0]))
pgrid

###############################################################################
# Generate the loop source field
# ------------------------------
#
# Here we generate the magnetic source field by making an electric square loop
# of 1 meter side length, hence an area of one square meter.

# Initiate a zero-valued source field.
sfield = emg3d.fields.SourceField(pgrid, freq=freq)

# Define the four dipole segments.
srcloop = [
    np.r_[src[0]-0.5, src[0]+0.5, src[1]-0.5, src[1]-0.5, src[2], src[2]],
    np.r_[src[0]+0.5, src[0]+0.5, src[1]-0.5, src[1]+0.5, src[2], src[2]],
    np.r_[src[0]+0.5, src[0]-0.5, src[1]+0.5, src[1]+0.5, src[2], src[2]],
    np.r_[src[0]-0.5, src[0]-0.5, src[1]+0.5, src[1]-0.5, src[2], src[2]],
]

# Add the source fields up.
for srcl in srcloop:
    sfield += emg3d.get_source_field(pgrid, srcl, freq, strength)


###############################################################################

# Get the model
pmodel = emg3d.Model(
        pgrid, property_x=resh, property_z=resv, mapping='Resistivity')

# Compute the electric field
efield = emg3d.solve(pgrid, pmodel, sfield, verb=3)

###############################################################################
# Compare the electric field generated from the magnetic source
# -------------------------------------------------------------
e3d_fs_ex = emg3d.get_receiver(pgrid, efield.fx, (rx, ry, zrec))
plot_result_rel(epm_fs_ex, e3d_fs_ex, x, r'Diffusive Fullspace $E_x$',
                vmin=-17, vmax=-10, mode='abs')

###############################################################################
e3d_fs_ey = emg3d.get_receiver(pgrid, efield.fy, (rx, ry, zrec))
plot_result_rel(epm_fs_ey, e3d_fs_ey, x, r'Diffusive Fullspace $E_y$',
                vmin=-17, vmax=-10, mode='abs')


###############################################################################
# Diffusive Fullspace :math:`E_z`
# -------------------------------
#
# The :math:`E_z`-field due to a z-directed magnetic source is zero.
#
# Compare the magnetic field generated from the magnetic source
# -------------------------------------------------------------
#
# Compute magnetic field :math:`H` from the electric field
# ````````````````````````````````````````````````````````

hfield = emg3d.get_h_field(pgrid, pmodel, efield)

###############################################################################
# Plot
# ````
e3d_fs_hx = emg3d.get_receiver(pgrid, hfield.fx, (rx, ry, zrec))
plot_result_rel(epm_fs_hx, e3d_fs_hx, x, r'Diffusive Fullspace $H_x$',
                vmin=-15, vmax=-8, mode='abs')

###############################################################################

e3d_fs_hy = emg3d.get_receiver(pgrid, hfield.fy, (rx, ry, zrec))
plot_result_rel(epm_fs_hy, e3d_fs_hy, x, r'Diffusive Fullspace $H_y$',
                vmin=-15, vmax=-8, mode='abs')

###############################################################################

e3d_fs_hz = emg3d.get_receiver(pgrid, hfield.fz, (rx, ry, zrec))
plot_result_rel(epm_fs_hz, e3d_fs_hz, x, r'Diffusive Fullspace $H_z$',
                vmin=-14, vmax=-7, mode='abs')

###############################################################################

plot_lineplot_ex(x, x, e3d_fs_hx.real, epm_fs_hx.real, pgrid)

###############################################################################
emg3d.Report()
