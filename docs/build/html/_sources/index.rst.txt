.. QPE_library documentation master file, created by
   sphinx-quickstart on Thu Feb 20 15:21:21 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*Rainforest* documentation!
=======================================


This python library provides a set of tools and lookup tables to

- Compute QPE maps using the RandomForest algorithm (:ref:`qpe module <qpe>`)
- Load, update and extract SQL queries on the gauge-radar database (:ref:`database module <db>`)
- Train the RandomForest algorithm using data from that database (:ref:`ml module <ml>`)
- Perform various operations on MeteoSwiss data such as read and plot file or retrieve COSMO or radar data from archive (:ref:`common module <common>`)

as well as a command line (:ref:`interface <Interface>`) to run most of the commands interactively.

.. note::
   On the CSCS, the source code is stored under */store/msrad/radar/rainforest/* and the database under */store/msrad/radar/radar_database/*



:ref:`Usage`
~~~~~~~~~~~~~~~~~

How to use the code on Kesch

:ref:`Reference`
~~~~~~~~~~~~~~~~~~~~~~~~

Comprehensive description of all the modules and functions available in the library.

:ref:`Examples`
~~~~~~~~~~~~~~~~~~~~~~~~~~

A few examples of use

Contents
========
.. toctree::
   :maxdepth: 1
   
   usage
   reference/index
   examples
   
