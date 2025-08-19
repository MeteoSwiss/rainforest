rainforest.database package
=================================

This module provides routines used for database management.

:mod:`rainforest.database.database` : main database class, used as an entry-point to the database

:mod:`rainforest.database.db\_populate` : command-line script used to add data to database

:mod:`rainforest.database.retrieve\_radar_data` : functions used to add new radar data to database

:mod:`rainforest.database.retrieve\_reference_data` : functions used to add new Cartesian reference data to database

:ref:`dwh` : R functions used to add new gauge data to database


.. _database_module:   
rainforest.database.database module
-----------------------------------------

.. automodule:: rainforest.database.database
   :members:
   :undoc-members:
   :show-inheritance:

rainforest.database.db\_populate module
---------------------------------------------

.. automodule:: rainforest.database.db_populate
   :members:
   :undoc-members:
   :show-inheritance:

rainforest.database.retrieve\_radar\_data module
------------------------------------------------------

.. automodule:: rainforest.database.retrieve_radar_data
   :members:
   :undoc-members:
   :show-inheritance:

rainforest.database.retrieve\_reference\_data module
----------------------------------------------------------

.. automodule:: rainforest.database.retrieve_reference_data
   :members:
   :undoc-members:
   :show-inheritance:

.. _dwh:

rainforest.database.retrieve\_dwh\_data R module
----------------------------------------------------------

Main routine for retrieving station data This is meant to be run as a command line command from a slurm script

i.e. Rscript retrieve_dwh_data.r <t0> <t1> <threshold> <stations> <variables> <output_folder> <missing_value overwrite>

IMPORTANT: this function is called by the main routine in database.py so you should never have to call it manually


**retrieve_dwh_data.R [options]**

Options:

-   **t0** (*str*)  - start time in YYYYMMDDHHMM format
-   **t1** (*str*)  - end time in YYYYMMDDHHMM format
-   **threshold** (*float*)  - minimum value of hourly precipitation for the entire hour to be included in the database (i.e. all 6 10min timesteps)
-   **variables** (*str*)  - list of variables to retrieve, using the DWH names, for example "tre200s0,prestas0,ure200s0,rre150z0,dkl010z0,fkl010z0"
-   **output_folder** (*str*)  -  directory where to store the csv files containing the retrieved data
-   **output_folder** (*float*)  -  directory where to store the csv files containing the retrieved data
-   **overwrite** (*bool*)  - whether or not to overwrite already existing data in the output\_folder
                        
