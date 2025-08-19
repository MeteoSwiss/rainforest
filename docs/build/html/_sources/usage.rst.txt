.. _Usage:

Usage
=======================================

Access library
--------------------------------------
On Kesch simply run the command

.. code-block:: console

    source activate radardb
    

Which will load the appropriate anaconda *virtualenv*, you can then directly use the command line tools (:ref:`interface`, :doc:`./reference/db_cmd`, :doc:`./reference/ml_cmd`, :doc:`./reference/qpe_cmd`)  in the standard console and/or import the library within python with


.. code-block:: python

    import rainforest
    
Note that on Kesch the radar database is stored in */store/msrad/radar/radar_database/*

Keeping database up-to-date
--------------------------------------

.. note:: 
    In the following, we expect that you are on Kesch you want to update the database but keep its current structure (same variables, same table names, same configuration). If you want something more fancy please read the doc carefully and decide by yourself.
    
To keep the database up-to-date you need to update the three tables *gauge*, *radar* and *reference*, always starting with the first one *gauge* as it is a prerequisite for the others. You can do this either by using the command-line  :ref:`interface` or by using the command :doc:`db_populate <./reference/db_cmd>`. 


.. code-block:: console

    db_populate -t gauge -s <start time in YYYYddmmHHMM format> -  <end time in YYYYddmmHHMM format> 
   
 
.. code-block:: console

    db_populate -t reference
    db_populate -t radar
 
Keeping randomForest model up-to-date
--------------------------------------

.. note:: 
    In the following, we expect that you are on Kesch and want to update only the *dual_pol* RF model in its default configuration and covering the whole timerange of the database. If you want something more fancy please read the doc carefully and decide by yourself.
    
Once you have updated the database you can use the command :doc:`rf_compute <./reference/ml_cmd>`.

.. code-block:: console

    rf_train -m 'dualpol_default'
    
