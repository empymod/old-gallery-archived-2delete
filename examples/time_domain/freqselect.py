"""
Interactive frequency selection
===============================

TODO
----

- Create an example using https://github.com/empymod/frequency-selection
- Update that repo with the version from the time-domain article and make a
  release.

----

Select the required frequency range and frequency density using ``empymod`` and
a layered model. The following parameters can be specified interactively:

- points per decade
- frequency range (min/max)
- offset
- Fourier transform (FFTLog or DLF with different filters)
- signal (impulse or switch-on/-off)

Other parameters have to be specified fix when initiating the widget.

"""
import emg3d

###############################################################################

emg3d.Report()
