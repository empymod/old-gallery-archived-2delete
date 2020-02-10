.. image:: https://raw.githubusercontent.com/empymod/emg3d-logo/master/logo-emg3d-cut.png
   :target: https://empymod.github.io
   :alt: emg3d logo
   
----

.. sphinx-inclusion-marker

Gallery for ``emg3d``. This is work in progress; the archived, old gallery can
be found in the repo `emg3d-examples
<https://github.com/empymod/emg3d-examples>`_.

Outstanding ToDo's:

- Adjust emg3d-manual.
- Create an environment.yml and scripts to run all.
- Create script to automatically deploy to gh-pages.
- Adjust READMEs at https://github.com/empymod/emg3d and here.
- Translate all notebooks [3/18 done]; still missing:

  - 1a_1D_VTI_empymod
  - 1b_2D_triaxial_MARE2DEM
  - 1c_3D_triaxial_SimPEG
  - 1d_1D_VTI_empymod-Laplace
  - 2a_SEG-EAGE_3D-Salt-Model
  - 3a_GemPy-discretize-emg3d
  - 4a_RAM-requirements
  - 4b_Runtime
  - 4c_Check_boundary4airwave
  - 5a_Obtaining_the_magnetic_field
  - 5b_Magnetic_permeability
  - 5c_Magnetic_source_using_el_loop
  - 5d_Magnetic_source_using_duality
  - 6a_Fullspace
  - 6b_Marine-1D


About ``emg3d``
===============

A multigrid solver for 3D electromagnetic diffusion with tri-axial electrical
anisotropy. The matrix-free solver can be used as main solver or as
preconditioner for one of the Krylov subspace methods implemented in
`scipy.sparse.linalg`, and the governing equations are discretized on a
staggered Yee grid. The code is written completely in Python using the
NumPy/SciPy-stack, where the most time- and memory-consuming parts are sped up
through jitted numba-functions.


More information
================

- **Website**: https://empymod.github.io,
- **Documentation**: https://emg3d.rtfd.io,
- **Source Code**: https://github.com/empymod/emg3d,


License information
===================

Copyright 2018-2020 The emg3d Developers.

Licensed under the Apache License, Version 2.0, see the ``LICENSE``-file.
