**************
Finding points
**************

The methods listed here let you find indices corresponding to points in time,
or perform aggregate functions (e.g. mean, min, max) on intervals within a time
series.

.. currentmodule:: datkit

Indices and values at specific times
====================================

.. autofunction:: index

.. autofunction:: index_near

.. autofunction:: index_on

.. autofunction:: index_crossing

.. autofunction:: time_crossing

.. autofunction:: value_at

.. autofunction:: value_near

.. autofunction:: value_interpolated

.. autofunction:: data_on

Averages and extremes
=====================

.. autofunction:: mean_on

.. autofunction:: max_on

.. autofunction:: min_on

.. autofunction:: abs_max_on

.. autofunction:: imax_on

.. autofunction:: imin_on

.. autofunction:: iabs_max_on

