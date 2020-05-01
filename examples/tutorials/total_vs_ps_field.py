r"""
Total vs primary/secondary field
================================

We usually use ``emg3d`` for total field calculations. However, we can also use
it in a primary-secondary field formulation, where we calculate the primary
field with a (semi-)analytical solution.

In this notebook we use ``emg3d`` to calculate

- Total field
- Primary field
- Secondary field

and compare the total field to the primary+secondary field.

The primary-field calculation could be replaced by a 1D modeller such as
``empymod``. You can play around with the required calculation-domain: Using a
primary-secondary formulation makes it possible to restrict the required
calculation domain for the scatterer a lot, therefore speeding up the
calculation. However, we do not dive into that in this notebook.

Background
----------

The total field is given by

.. math::
    :label: totalfield

    s \mu \sigma \mathbf{\hat{E}} + \nabla \times
    \nabla \times \mathbf{\hat{E}} =
    -s\mu\mathbf{\hat{J}}_s .

We can split it up into a primary field :math:`\mathbf{\hat{E}}^p` and a
secondary field :math:`\mathbf{\hat{E}}^s`,

.. math::
    :label: fieldsplit

    \mathbf{\hat{E}} =  \mathbf{\hat{E}}^p + \mathbf{\hat{E}}^s,

where we also have to split our conductivity model into

.. math::
    :label: modelsplit

    \sigma = \sigma^p + \Delta\sigma.

The primary field could be just the direct field, or the direct field plus the
air layer, or an entire 1D background, something that can be calculated
(semi-)analytically. The secondary field is everything that is not included in
the primary field.

The primary field is then given by

.. math::
    :label: primaryfield

    s \mu \sigma^p \mathbf{\hat{E}}^p + \nabla \times
    \nabla \times \mathbf{\hat{E}}^p =
    -s\mu\mathbf{\hat{J}}_s ,

and the secondary field can be calculated using the primary field as source,

.. math::
    :label: secondaryfield

    s \mu \sigma \mathbf{\hat{E}}^s + \nabla \times
    \nabla \times \mathbf{\hat{E}}^s =
    -s\mu\Delta\sigma\mathbf{\hat{E}}^p .

"""
import emg3d
import discretize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.style.use('ggplot')
# sphinx_gallery_thumbnail_number = 3

###############################################################################
# Survey
# ------

src = [0, 0, -950, 0, 0]    # x-dir. source at the origin, 50 m above seafloor
off = np.arange(5, 81)*100  # Offsets
rec = [off, off*0, -1000]   # In-line receivers on the seafloor
res = [1e10, 0.3, 1]        # Resistivities (Hz): [air, seawater, background]
freq = 1.0                  # Frequency (Ohm.m)

###############################################################################
# Mesh
# ----
#
# We create quite a coarse grid (100 m minimum cell width), to have reasonable
# fast calculation times.
#
# Also note that the mesh here includes a large boundary because of the air
# layer. If you use a semi-analytical solution for the 1D background you could
# restrict that domain a lot.

meshinp = {'freq': freq, 'min_width': 100, 'verb': 0}
xx, x0 = emg3d.meshes.get_hx_h0(
    res=[res[1], 100.], fixed=src[0], domain=[-100, 8100], **meshinp)
yy, y0 = emg3d.meshes.get_hx_h0(
    res=[res[1], 100.], fixed=src[1], domain=[-500, 500], **meshinp)
zz, z0 = emg3d.meshes.get_hx_h0(
    res=[res[1], res[2], 100.], domain=[-2500, 0], fixed=[-1000, 0, -2000],
    **meshinp)

grid = discretize.TensorMesh([xx, yy, zz], x0=np.array([x0, y0, z0]))
grid

###############################################################################
# Create model
# ------------

# Layered_background
res_x = np.ones(grid.nC)*res[0]            # Air resistivity
res_x[grid.gridCC[:, 2] < 0] = res[1]      # Water resistivity
res_x[grid.gridCC[:, 2] < -1000] = res[2]  # Background resistivity

# Background model
model_pf = emg3d.models.Model(grid, res_x.copy())

# Include the target
xx = (grid.gridCC[:, 0] >= 0) & (grid.gridCC[:, 0] <= 6000)
yy = abs(grid.gridCC[:, 1]) <= 500
zz = (grid.gridCC[:, 2] > -2500)*(grid.gridCC[:, 2] < -2000)

res_x[xx*yy*zz] = 100.  # Target resistivity

# Create target model
model = emg3d.models.Model(grid, res_x)

# Plot a slice
grid.plot_3d_slicer(
        model.res_x, zslice=-2250, clim=[0.3, 200],
        xlim=(-1000, 8000), ylim=(-4000, 4000), zlim=(-3000, 500),
        pcolorOpts={'norm': LogNorm()}
)

###############################################################################
# Calculate total field with ``emg3d``
# ------------------------------------

modparams = {
        'verb': -1, 'sslsolver': True,
        'semicoarsening': True, 'linerelaxation': True
}

sfield_tf = emg3d.fields.get_source_field(grid, src, freq, strength=0)
em3_tf = emg3d.solve(grid, model, sfield_tf, **modparams)


###############################################################################
# Calculate primary field (1D background) with ``emg3d``
# ------------------------------------------------------
#
# Here we use ``emg3d`` to calculate the primary field. This could be replaced
# by a (semi-)analytical solution.

sfield_pf = emg3d.fields.get_source_field(grid, src, freq, strength=0)
em3_pf = emg3d.solve(grid, model_pf, sfield_pf, **modparams)

###############################################################################
# Calculate secondary field (scatterer) with ``emg3d``
# ----------------------------------------------------
#
# Define the secondary source
# ```````````````````````````

# Get the difference of conductivity as volume-average values
dsigma = grid.vol.reshape(grid.vnC, order='F')*(1/model.res_x-1/model_pf.res_x)

# Here we use the primary field calculated with emg3d. This could be done
# with a 1D modeller such as empymod instead.
fx = em3_pf.fx.copy()
fy = em3_pf.fy.copy()
fz = em3_pf.fz.copy()

# Average delta sigma to the corresponding edges
fx[:, 1:-1, 1:-1] *= 0.25*(dsigma[:, :-1, :-1] + dsigma[:, 1:, :-1] +
                           dsigma[:, :-1, 1:] + dsigma[:, 1:, 1:])
fy[1:-1, :, 1:-1] *= 0.25*(dsigma[:-1, :, :-1] + dsigma[1:, :, :-1] +
                           dsigma[:-1, :, 1:] + dsigma[1:, :, 1:])
fz[1:-1, 1:-1, :] *= 0.25*(dsigma[:-1, :-1, :] + dsigma[1:, :-1, :] +
                           dsigma[:-1, 1:, :] + dsigma[1:, 1:, :])

# Create field instance iwu dsigma E
sfield_sf = sfield_pf.smu0*emg3d.fields.Field(fx, fy, fz, freq=freq)
sfield_sf.ensure_pec

###############################################################################
# Plot the secondary source
# `````````````````````````
#
# Our secondary source is the entire target, the scatterer. Here we look at the
# :math:`E_x` secondary source field. But note that the secondary source has
# all three components :math:`E_x`, :math:`E_y`, and :math:`E_z`, even though
# our primary source was purely :math:`x`-directed. (Change ``fx`` to ``fy`` or
# ``fz`` in the command below, and simultaneously ``Ex`` to ``Ey`` or ``Ez``,
# to show the other source fields.)

grid.plot_3d_slicer(
        sfield_sf.fx.ravel('F'), view='abs', vType='Ex',
        zslice=-2250, clim=[1e-17, 1e-9],
        xlim=(-1000, 8000), ylim=(-4000, 4000), zlim=(-3000, 500),
        pcolorOpts={'norm': LogNorm()}
)

###############################################################################
# Calculate the secondary source
# ``````````````````````````````

em3_sf = emg3d.solve(grid, model, sfield_sf, **modparams)

###############################################################################
# Plot result
# -----------

# E = E^p + E^s
em3_ps = em3_pf + em3_sf

# Get the responses at receiver locations
rectuple = (rec[0], rec[1], rec[2])
em3_pf_rec = emg3d.fields.get_receiver(grid, em3_pf.fx, rectuple)
em3_tf_rec = emg3d.fields.get_receiver(grid, em3_tf.fx, rectuple)
em3_sf_rec = emg3d.fields.get_receiver(grid, em3_sf.fx, rectuple)
em3_ps_rec = emg3d.fields.get_receiver(grid, em3_ps.fx, rectuple)

###############################################################################
plt.figure(figsize=(9, 5))

ax1 = plt.subplot(121)
plt.title('|Real part|')
plt.plot(off/1e3, abs(em3_pf_rec.real), 'k',
         label='Primary Field (1D Background)')
plt.plot(off/1e3, abs(em3_sf_rec.real), '.4', ls='--',
         label='Secondary Field (Scatterer)')
plt.plot(off/1e3, abs(em3_ps_rec.real))
plt.plot(off[::2]/1e3, abs(em3_tf_rec[::2].real), '.')
plt.plot(off/1e3, abs(em3_ps_rec.real-em3_tf_rec.real))
plt.xlabel('Offset (km)')
plt.ylabel('$E_x$ (V/m)')
plt.yscale('log')
plt.legend()

ax2 = plt.subplot(122, sharey=ax1)
plt.title('|Imaginary part|')
plt.plot(off/1e3, abs(em3_pf_rec.imag), 'k')
plt.plot(off/1e3, abs(em3_sf_rec.imag), '.4', ls='--')
plt.plot(off/1e3, abs(em3_ps_rec.imag), label='P/S Field')
plt.plot(off[::2]/1e3, abs(em3_tf_rec[::2].imag), '.', label='Total Field')
plt.plot(off/1e3, abs(em3_ps_rec.imag-em3_tf_rec.imag),
         label=r'$\Delta$|P/S-Total|')
plt.xlabel('Offset (km)')
plt.ylabel('$E_x$ (V/m)')
plt.yscale('log')
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
plt.legend()

plt.tight_layout()
plt.show()

###############################################################################

emg3d.Report([discretize, ])
