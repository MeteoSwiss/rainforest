"""
This submodule takes care of database interfacing and updating. It provides
the following files:
    
-   database.py : main Database class, interface to add new data and perform
    queries
-   retrieve_radar_data.py : functions to process polar data and add it 
    to the database
-   retrieve_radar_data.r : functions to process station data and add it 
    to the database
-   retrieve_reference_data.py : functions to process Cartesian data and add it 
    to the database
-   db_populate.py : command-line tool to update the database
-   default_config.yml : default configuration file for database updating
"""

from .database import Database