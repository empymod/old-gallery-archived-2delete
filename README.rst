.. image:: https://raw.githubusercontent.com/empymod/emg3d-logo/master/logo-emg3d-cut.png
   :target: https://empymod.github.io
   :alt: emg3d logo
   
----

.. sphinx-inclusion-marker


This will become the emg3d-gallery, and replace the emg3d-examples.

TODO:

- [ ] Translate all notebooks.
- [ ] Mark them accordingly as outdated in emg3d-examples.
- [ ] Adjust emg3d-manual.
- [ ] Create an environment.yml and scripts to run all.
- [ ] Create command to publish to gh-pages.
- [ ] Adjust README.


A multigrid solver for 3D electromagnetic diffusion with tri-axial electrical
anisotropy. The matrix-free solver can be used as main solver or as
preconditioner for one of the Krylov subspace methods implemented in
`scipy.sparse.linalg`, and the governing equations are discretized on a
staggered Yee grid. The code is written completely in Python using the
NumPy/SciPy-stack, where the most time- and memory-consuming parts are sped up
through jitted numba-functions.


More information
================
For more information regarding installation, usage, contributing, roadmap, bug
reports, and much more, see

- **Website**: https://empymod.github.io,
- **Documentation**: https://emg3d.rtfd.io,
- **Source Code**: https://github.com/empymod/emg3d,
- **Examples**: https://emg3d.rtfd.io/en/stable/examples.


License information
===================

Copyright 2018-2020 The emg3d Developers.

Licensed under the Apache License, Version 2.0, see the ``LICENSE``-file.
