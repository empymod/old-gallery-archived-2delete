"""
1. Minimum working example
==========================

This is a simple minimum working example to use the multigrid solver `emg3d`,
along the lines of the one provided in the manual as `"Basic Example"
<https://emg3d.readthedocs.io/en/stable/usage.html#basic-example>`_.

To see some more realistic computations have a look at the other examples in
this gallery. In particularly at
:ref:`sphx_glr_gallery_tutorials_simulation.py` to see how to use `emg3d` for a
complex survey with many sources and frequencies.

This example uses advanced tools of meshing including plotting, for which you
need to install additionally ``discretize`` and ``matplotlib``. If you are
interested in a basic example that only requires ``emg3d`` and its mandatory
dependencies here it is:

.. code-block:: python

    import emg3d
    import numpy as np

    # Create a simple grid, 8 cells of length 1 in each direction,
    # starting at the origin.
    hx = np.ones(8)
    grid = emg3d.TensorMesh(h=[hx, hx, hx], x0=np.array([0, 0, 0]))

    # The model is a fullspace with tri-axial anisotropy.
    model = emg3d.Model(grid=grid, property_x=1.5, property_y=1.8,
                        property_z=3.3, mapping='Resistivity')

    # The source is an x-directed, horizontal dipole at (4, 4, 4),
    # frequency is 10 Hz.
    sfield = emg3d.get_source_field(grid=grid, src=[4, 4, 4, 0, 0], freq=10.0)

    # Compute the electric signal.
    efield = emg3d.solve(grid=grid, model=model, sfield=sfield, verb=3)

    # Get the corresponding magnetic signal.
    hfield = emg3d.get_h_field(grid=grid, model=model, field=efield)


Let's start by loading the required modules:
"""
import emg3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.style.use('ggplot')
# sphinx_gallery_thumbnail_number = 2

###############################################################################
# 1. Mesh
# -------
#
# First we define the mesh (see :class:`emg3d.meshes.TensorMesh` for more
# info). In reality, this task requires some careful considerations. E.g., to
# avoid edge effects, the mesh should be large enough in order for the fields
# to dissipate, yet fine enough around source and receiver to accurately model
# them. This grid is too small by any means, but serves as a minimal example.
# (Note that in order to define a mesh in such a way you must have `discretize`
# installed; see above for more info.)

grid = emg3d.TensorMesh(
        [[(25, 10, -1.04), (25, 28), (25, 10, 1.04)],
         [(50, 8, -1.03), (50, 16), (50, 8, 1.03)],
         [(30, 8, -1.05), (30, 16), (30, 8, 1.05)]],
        x0='CCC')
grid

###############################################################################
# 2. Model
# --------
#
# Next we define a very simple fullspace resistivity model with
# :math:`\rho_x=1.5\,\Omega\,\rm{m}`, :math:`\rho_y=1.8\,\Omega\,\rm{m}`, and
# :math:`\rho_z=3.3\,\Omega\,\rm{m}`.

model = emg3d.Model(grid, property_x=1.5, property_y=1.8,
                    property_z=3.3, mapping='Resistivity')

###############################################################################
# Here we define the model in terms of resistivity. Have a look at the example
# :ref:`sphx_glr_gallery_tutorials_mapping.py` to see how to define models
# in terms of conductivity or their logarithms.
#
# Plotting this model results in an obviously rather boring plot, as it simply
# shows a homogeneous space. Here we plot the x-directed resistivity:

grid.plot_3d_slicer(np.ones(grid.vnC)*model.property_x)  # x-resistivity

###############################################################################
# 3. Source field
# ---------------
#
# The source is an x-directed dipole at the origin, with a 10 Hz signal of 1 A
# (``src`` is defined either as ``[x, y, z, dip, azimuth]`` or ``[x0, x1, y0,
# y1, z0, z1]``; the strength can be set via the ``strength`` parameter).

sfield = emg3d.get_source_field(grid=grid, src=[0, 0, 0, 0, 0], freq=10)

###############################################################################
# 4. Compute the electric field
# -----------------------------
#
# Finally we can compute the electric field with ``emg3d``:

efield = emg3d.solve(grid=grid, model=model, sfield=sfield, verb=3)

###############################################################################
# The computation requires in this case seven multigrid F-cycles and takes just
# a few seconds. It was able to coarsen in each dimension four times, where the
# input grid had 49,152 cells, and the coarsest grid had 12 cells.
#
# 5. Plot the result
# ------------------
#
# We can again utilize the in-built functions of a ``discretize``-grid to plot,
# e.g., the x-directed electric field.

grid.plot_3d_slicer(
        efield.fx.ravel('F'), view='abs', v_type='Ex',
        pcolor_opts={'norm': LogNorm()}
)


###############################################################################
# 6. Compute and plot the magnetic field
# --------------------------------------
#
# We can also get the magnetic field and plot it (note that `v_type='Fx'` now,
# not `Ex`, as the magnetic fields lives on the faces of the Yee grid):

hfield = emg3d.get_h_field(grid=grid, model=model, field=efield)
grid.plot_3d_slicer(
        hfield.fx.ravel('F'), view='abs', v_type='Fx',
        pcolor_opts={'norm': LogNorm()}
)

###############################################################################

emg3d.Report()
