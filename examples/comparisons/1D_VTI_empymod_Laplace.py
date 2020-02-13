r"""
empymod: 1D VTI Laplace-domain
==============================

1D VTI comparison between ``emg3d`` and ``empymod`` in the Laplace domain.

The code ``empymod`` is an open-source code which can model CSEM responses for
a layered medium including VTI electrical anisotropy, see `empymod.github.io
<https://empymod.github.io>`_.

Content:

1. Full-space VTI model for a finite length, finite strength, rotated bipole.
2. Layered model for a deep water model with a point dipole source.


Both codes, ``empymod`` and ``emg3d``, are able to calculate the EM response in
the Laplace domain, by using a real value :math:`s` instead of the complex
value :math:`\mathrm{i}\omega=2\mathrm{i}\pi f`. To calculate the response in
the Laplace domain in the two codes you have to provide negative values for the
``freq``-parameter, which are then considered ``s-value``.

"""
import emg3d
import empymod
import discretize
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as sint
from matplotlib.colors import LogNorm
plt.style.use('ggplot')
# sphinx_gallery_thumbnail_path = '_static/thumbs/empymod-s.png'


###############################################################################
# Plotting and interpolation routines
# -----------------------------------
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map
    https://gist.github.com/jakevdp/91077b0cae40f8f8244a
    """
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


###############################################################################
def plot_data_rel(ax, name, data, x, vmin=-15., vmax=-7., error=False):
    """Plot function."""

    ax.set_title(name)
    ax.set_xlim(min(x)/1000, max(x)/1000)
    ax.set_ylim(min(x)/1000, max(x)/1000)
    ax.axis("equal")

    if error:
        cf = ax.pcolormesh(
                x/1000, x/1000, np.log10(data), vmin=vmin, vmax=vmax,
                linewidth=0, rasterized=True, cmap=discrete_cmap(8, "RdBu_r"))
    else:
        cf = ax.pcolormesh(
                x/1000, x/1000, np.log10(np.abs(data)), linewidth=0,
                rasterized=True, cmap="viridis", vmin=vmin, vmax=vmax)

    return cf


###############################################################################
def plot_result_rel(depm, de3d, x, title, vmin=-15., vmax=-7.):
    fig, axs = plt.subplots(figsize=(18, 7.5), nrows=1, ncols=3)

    # Plot Re(data)
    cf0 = plot_data_rel(
            axs[0], r"(a) |empymod|", depm.real, x, vmin, vmax, error=False)
    plot_data_rel(
            axs[1], r"(b) |emg3d|", de3d.real, x, vmin, vmax, error=False)
    cf2 = plot_data_rel(
            axs[2], r"(c) Error", np.abs((depm.real-de3d.real)/depm.real)*100,
            x, vmin=-2, vmax=2, error=True)

    # Colorbars
    fig.colorbar(cf0, ax=axs[:2], label=r"$\log_{10}$ Amplitude (V/m)",
                 aspect=40, orientation='horizontal')
    cbar = fig.colorbar(cf2, ax=axs[2], label=r"Relative Error",
                        orientation='horizontal')
    cbar.set_ticks([-2, -1, 0, 1, 2])
    cbar.ax.set_yticklabels([r"$0.01\,\%$", r"$0.1\,\%$", r"$1\,\%$",
                             r"$10\,\%$", r"$100\,\%$"])

    # Axis label
    axs[0].set_ylabel("Crossline Offset (km)")
    axs[1].set_xlabel("Inline Offset (km)")

    # Title
    fig.suptitle(title, y=1, fontsize=20)
    plt.show()


###############################################################################
def plot_lineplot_ex(x, y, data, epm_fs, grid):
    xi = x.size//2
    yi = y.size//2

    fn = sint.interp1d(x, data[:, xi], bounds_error=False)
    x1 = fn(grid.vectorNx)

    fn = sint.interp1d(y, data[yi, :], bounds_error=False)
    y1 = fn(grid.vectorNx)

    plt.figure(figsize=(15, 8))

    plt.plot(x/1e3, np.abs(epm_fs[:, xi]), 'C0', lw=3, label='Inline empymod')
    plt.plot(x/1e3, np.abs(data[:, xi]), 'k--', label='Inline emg3d')
    plt.plot(grid.vectorNx/1e3, np.abs(x1), 'k*')

    plt.plot(y/1e3, np.abs(epm_fs[yi, :]), 'C1', lw=3,
             label='Crossline empymod')
    plt.plot(y/1e3, np.abs(data[yi, :]), 'k:', label='Crossline emg3d')
    plt.plot(grid.vectorNx/1e3, np.abs(y1), 'k*', label='Grid points emg3d')

    plt.yscale('log')
    plt.title(r'Inline and crossline $E_x$', fontsize=20)
    plt.xlabel('Offset (km)', fontsize=14)
    plt.ylabel(r'|Amplitude (V/m)|', fontsize=14)
    plt.legend()
    plt.show()


###############################################################################
# 1. Full-space VTI model for a finite length, finite strength, rotated bipole
# ----------------------------------------------------------------------------
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
src = [-50, 50, -30, 30, -320., -280.]  # Source: [x1, x2, y1, y2, z1, z2]
src_c = np.mean(np.array(src).reshape(3, 2), 1).ravel()  # Center pts of source
zrec = -400.           # Receiver depth
sval = -4.84           # Laplace-value
strength = np.pi       # Source strength

# Input for empymod
model = {  # empymod has positive z-down, so switch source-z
    'src': [src[0], src[1], src[2], src[3], -src[4], -src[5]],
    'depth': [],
    'res': resh,
    'aniso': aniso,
    'strength': strength,
    'srcpts': 5,
    'freqtime': sval,
    'htarg': {'pts_per_dec': -1},
}


###############################################################################

epm_fs_x = empymod.bipole(rec=[rx.ravel(), ry.ravel(), -zrec, 0, 0], verb=3,
                          **model).reshape(np.shape(rx))
epm_fs_y = empymod.bipole(rec=[rx.ravel(), ry.ravel(), -zrec, 90, 0], verb=1,
                          **model).reshape(np.shape(rx))
epm_fs_z = empymod.bipole(rec=[rx.ravel(), ry.ravel(), -zrec, 0, -90], verb=1,
                          **model).reshape(np.shape(rx))

###############################################################################
# emg3d
# `````

# Get calculation domain as a function of frequency (resp., skin depth)
hx_min, xdomain = emg3d.utils.get_domain(x0=src[0], freq=0.1, min_width=20)
hz_min, zdomain = emg3d.utils.get_domain(x0=src[2], freq=0.1, min_width=20)

# Create stretched grid
nx = 2**7
hx = emg3d.utils.get_stretched_h(hx_min, xdomain, nx, src_c[0])
hy = emg3d.utils.get_stretched_h(hx_min, xdomain, nx, src_c[1])
hz = emg3d.utils.get_stretched_h(hz_min, zdomain, nx, src_c[2])
pgrid = discretize.TensorMesh([hx, hy, hz],
                              x0=(xdomain[0], xdomain[0], zdomain[0]))
pgrid


###############################################################################

# Get the model
pmodel = emg3d.utils.Model(pgrid, res_x=resh, res_z=resv)

# Get the source field
sfield = emg3d.utils.get_source_field(pgrid, src, sval, strength)

# Calculate the electric field
pfield = emg3d.solve(pgrid, pmodel, sfield, verb=3)


###############################################################################
# Plot
# ````

e3d_fs_x = emg3d.utils.get_receiver(pgrid, pfield.fx, (rx, ry, zrec))
plot_result_rel(epm_fs_x, e3d_fs_x, x, r'Diffusive Fullspace $E_x$',
                vmin=-12, vmax=-6)


###############################################################################

e3d_fs_y = emg3d.utils.get_receiver(pgrid, pfield.fy, (rx, ry, zrec))
plot_result_rel(epm_fs_y, e3d_fs_y, x, r'Diffusive Fullspace $E_y$',
                vmin=-12, vmax=-6)


###############################################################################

e3d_fs_z = emg3d.utils.get_receiver(pgrid, pfield.fz, (rx, ry, zrec))
plot_result_rel(epm_fs_z, e3d_fs_z, x, r'Diffusive Fullspace $E_z$',
                vmin=-12, vmax=-6)


###############################################################################

plot_lineplot_ex(x, x, e3d_fs_x.real, epm_fs_x.real, pgrid)


###############################################################################
# 2. Layered model for a deep water model with a point dipole source
# ------------------------------------------------------------------
#
# empymod
# ```````

# Survey parameters
x = (np.arange(1025))*5-2560
rx = np.repeat([x, ], np.size(x), axis=0)
ry = rx.transpose()

# Model parameters
resh = [1e12, 0.3, 1, 50, 1]      # Horizontal resistivity
aniso = np.sqrt([1, 1, 2, 2, 2])  # Anisotropy
resv = resh*aniso**2              # Vertical resistivity
src = [0, 0, -900, 0, 0]          # Source: [x, y, z, azimuth, dip]
zrec = -950.                      # Receiver depth
sval = -2*np.pi                   # Laplace-value
depth = np.array([0, -1000, -2000, -2200])  # Layer boundaries

model = {  # empymod has positive z-down, so switch source-z
    'src': [src[0], src[1], -src[2], src[3], src[4]],
    'depth': -depth,
    'res': resh,
    'aniso': aniso,
    'freqtime': sval,
    'htarg': {'pts_per_dec': -1},
}


###############################################################################

epm_deep_x = empymod.bipole(rec=[rx.ravel(), ry.ravel(), -zrec, 0, 0],
                            verb=3, **model).reshape(np.shape(rx))
epm_deep_y = empymod.bipole(rec=[rx.ravel(), ry.ravel(), -zrec, 90, 0],
                            verb=1, **model).reshape(np.shape(rx))
epm_deep_z = empymod.bipole(rec=[rx.ravel(), ry.ravel(), -zrec, 0, -90],
                            verb=1, **model).reshape(np.shape(rx))

###############################################################################
# emg3d
# `````


# Get calculation domain as a function of frequency (resp., skin depth)
hx_min, xdomain = emg3d.utils.get_domain(x0=src[0], freq=0.1, min_width=20)
hz_min, zdomain = emg3d.utils.get_domain(
        x0=src[2], freq=0.1, min_width=20, fact_pos=10)

# Create stretched grid
nx = 2**7
hx = emg3d.utils.get_stretched_h(hx_min, xdomain, nx, src[0])
hy = emg3d.utils.get_stretched_h(hx_min, xdomain, nx, src[1])
hz = emg3d.utils.get_stretched_h(hz_min, zdomain, nx*2, x0=depth[-1], x1=0)
pgrid = discretize.TensorMesh(
        [hx, hy, hz], x0=(xdomain[0], xdomain[0], zdomain[0]))
pgrid


###############################################################################

# Create the model: horizontal resistivity
res_x_full = resh[-1]*np.ones(pgrid.nC)  # Background
res_x_full[pgrid.gridCC[:, 2] >= depth[3]] = resh[3]  # Target
res_x_full[pgrid.gridCC[:, 2] >= depth[2]] = resh[2]  # Overburden
res_x_full[pgrid.gridCC[:, 2] >= depth[1]] = resh[1]  # Water
res_x_full[pgrid.gridCC[:, 2] >= depth[0]] = resh[0]  # Air

# Create the model: vertical resistivity
res_z_full = resv[-1]*np.ones(pgrid.nC)  # Background
res_z_full[pgrid.gridCC[:, 2] >= depth[3]] = resv[3]
res_z_full[pgrid.gridCC[:, 2] >= depth[2]] = resv[2]
res_z_full[pgrid.gridCC[:, 2] >= depth[1]] = resv[1]
res_z_full[pgrid.gridCC[:, 2] >= depth[0]] = resv[0]

# Get the model
pmodel = emg3d.utils.Model(pgrid, res_x_full, res_z=res_z_full)

# Plot it
pgrid.plot_3d_slicer(pmodel.res_x, zslice=-2000, clim=[0.3, 50],
                     zlim=(-5000, 50), pcolorOpts={'norm': LogNorm()})


###############################################################################

# Get the source field
sfield = emg3d.utils.get_source_field(pgrid, src, sval, 0)

# Calculate the electric field
pfield = emg3d.solve(pgrid, pmodel, sfield, verb=3)


###############################################################################
# Plot
# ````

e3d_deep_x = emg3d.utils.get_receiver(pgrid, pfield.fx, (rx, ry, zrec))
plot_result_rel(epm_deep_x, e3d_deep_x, x, r'Deep water point dipole $E_x$',
                vmin=-14, vmax=-8)


###############################################################################

e3d_deep_y = emg3d.utils.get_receiver(pgrid, pfield.fy, (rx, ry, zrec))
plot_result_rel(epm_deep_y, e3d_deep_y, x, r'Deep water point dipole $E_y$',
                vmin=-14, vmax=-8)


###############################################################################

e3d_deep_z = emg3d.utils.get_receiver(pgrid, pfield.fz, (rx, ry, zrec))
plot_result_rel(epm_deep_z, e3d_deep_z, x, r'Deep water point dipole $E_z$',
                vmin=-14, vmax=-8)


###############################################################################

plot_lineplot_ex(x, x, e3d_deep_x.real, epm_deep_x.real, pgrid)


###############################################################################

emg3d.Report(empymod)
