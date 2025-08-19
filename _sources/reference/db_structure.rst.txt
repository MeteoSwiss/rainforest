Database structure
==========================================

The database is divided into three tables

#. :ref:`gaugetab` table which contains data from the SwissMetNet stations and pluviometers
#. :ref:`reftab` table which contains Cartesian data from MeteoSwiss such as CPC, RZC, MZC, at the location of the stations
#. :ref:`radtab` table which contains polar radar data interpolated above the stations

Currently these tables are stored in */store/msrad/radar/radar_database/* with a subfolder for every table. Note that the tables are spread over many files, namely one single  `parquet <https://fr.wikipedia.org/wiki/Apache_Parquet>`_ file for every different day in the *reference* and *radar* tables and one single GZIP compressed CSV file for every SwissMetNet/Pluvio station in the *gauge* table. These files can easily be read in batch as a single dataframe using either `pyspark <https://spark.apache.org/docs/latest/api/python/index.html>`_ or `dask <https://dask.org/>`_

All tables share two attributes **STATION** (the identifier of the SwissMetNet/Pluvio) and **TIMESTAMP** (Unix timestamp), together they form a identifier that can be used to join the tables using :doc:`SQL queries <interface>` or the *query* command in the *Database* class.

.. _gaugetab:

*Gauge table*
-----------------

.. csv-table:: Gauge
   :file: gauge.csv
   :widths: 10, 80, 10
   :header-rows: 1
   
   
.. _reftab:   

*Reference table*
-----------------
.. csv-table:: Reference
   :file: reference.csv
   :widths: 10, 80, 10
   :header-rows: 1
   
  
   
.. _radtab:   

*Radar table*
-----------------
In the radar table if different :doc:`aggregation methods <db_options>` have been used, a given variable will be present multiple times with different suffixes, i.e. *ZH\_mean*, *ZH\_max*.

.. csv-table:: Radar
   :file: radar.csv
   :widths: 10,10, 80, 10
   :header-rows: 1
   
   
