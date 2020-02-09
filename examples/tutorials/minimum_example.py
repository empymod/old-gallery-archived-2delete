"""
Minimum working example
=======================

This is a simple minimum working example to get started, along the lines of the
one provided in the manual as `"Basic Example"
<https://emg3d.readthedocs.io/en/stable/usage.html#basic-example>`_.

To see some more realistic models have a look at the other examples in this
gallery.

This notebooks uses :class:`discretize.TensorMesh` to create meshes easily and
plot the model as well as the resulting electric field, which also requires
``matplotlib``. If you are interested in a basic example that only requires
``emg3d`` and its mandatory dependencies here it is:

.. code-block:: python

    import emg3d
    import numpy as np

    # Create a simple grid, 8 cells of length 1 in each direction,
    # starting at the origin.
    grid = emg3d.utils.TensorMesh(
            [np.ones(8), np.ones(8), np.ones(8)], x0=np.array([0, 0, 0]))

    # The model is a fullspace with tri-axial anisotropy.
    model = emg3d.utils.Model(grid, res_x=1.5, res_y=1.8, res_z=3.3)

    # The source is a x-directed, horizontal dipole at (4, 4, 4),
    # frequency is 10 Hz.
    sfield = emg3d.utils.get_source_field(grid, src=[4, 4, 4, 0, 0], freq=10.0)

    # Calculate the electric signal.
    efield = emg3d.solver.solver(grid, model, sfield, verb=3)

    # Get the corresponding magnetic signal.
    hfield = emg3d.utils.get_h_field(grid, model, efield)

Note that in the future ``discretize`` is likely to become a mandatory
dependency as well, which will make above mini-example obsolete.

The first step is to load ``emg3d`` and ``discretize`` (to create a mesh),
along with ``numpy`` and ``matplotlib``:

"""
import emg3d
import discretize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.style.use('ggplot')
# sphinx_gallery_thumbnail_number = 2

###############################################################################
# 1. Mesh
# -------
#
# We start by defining the mesh (see class:`discretize.TensorMesh` for more
# info). In reality, this task requires some careful considerations. E.g., to
# avoid edge effects, the mesh should be large enough in order for the fields
# to dissipate, yet fine enough around source and receiver to accurately model
# them. This grid is too small by any means, but serves as a minimal example.

grid = discretize.TensorMesh(
        [[(25, 10, -1.04), (25, 28), (25, 10, 1.04)],
         [(50, 8, -1.03), (50, 16), (50, 8, 1.03)],
         [(30, 8, -1.05), (30, 16), (30, 8, 1.05)]],
        x0='CCC')
grid

###############################################################################
# 2. Model
# --------
#
# Next we define a very simple fullspace model with
# :math:`\rho_x=1.5\,\Omega\,\rm{m}`, :math:`\rho_y=1.8\,\Omega\,\rm{m}`, and
# :math:`\rho_z=3.3\,\Omega\,\rm{m}`.

model = emg3d.utils.Model(grid, res_x=1.5, res_y=1.8, res_z=3.3)

###############################################################################
# We can plot the model using ``discretize``; in this case it is obviously
# a rather boring plot, as it simply shows a homogeneous space.

grid.plot_3d_slicer(np.ones(grid.vnC)*model.res_x)  # x-resistivity

###############################################################################
# 3. Source field
# ---------------
#
# The source is an x-directed dipole at the origin, with a 10 Hz signal of 1 A
# (``src`` is defined either as ``[x, y, z, dip, azimuth]`` or ``[x0, x1, y0,
# y1, z0, z1]``; the strength can be set via the ``strength`` parameter).

sfield = emg3d.utils.get_source_field(grid, src=[0, 0, 0, 0, 0], freq=10)

###############################################################################
# 4. Calculate the electric field
# -------------------------------
#
# Now we can calculate the electric field with ``emg3d``:

efield = emg3d.solver.solver(grid, model, sfield, verb=3)

###############################################################################
# The calculation requires in this case seven multigrid F-cycles and takes just
# a few seconds. It was able to coarsen in each dimension four times, where the
# input grid had 49,152 cells, and the coarsest grid had 12 cells.
#
# 5. Plot the result
# ------------------
#
# We can again utilize the in-built functions of a ``discretize``-grid to plot,
# e.g., the x-directed electric field.

grid.plot_3d_slicer(
        efield.fx.ravel('F'), view='abs', vType='Ex',
        pcolorOpts={'norm': LogNorm()}
)

###############################################################################

emg3d.Report()
