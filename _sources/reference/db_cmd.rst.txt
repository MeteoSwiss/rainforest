Database command-line tool
==========================================

The *database* submodule has only a command-line tool to populate (update) the database. For SQL queries please use the command-line :ref:`Interface`.

.. _db_populate:

*db_populate*
-----------------

Updates any of the three tables of the database *gauge*, *radar* and *reference* with new data. 

**db_populate [options]**

Options:
  -h, --help            show this help message and exit
  -t TYPE, --type=TYPE  Type of table to populate, either 'gauge', 'reference'
                        or 'radar'
  -o OUTPUT, --outputfolder=OUTPUT
                        Path of the output folder, default is
                        /store/msrad/radar/radar_database/<type>
  -s START, --start=START
                        Specify the start time in the format YYYYddmmHHMM, it
                        is mandatory only if type == 'gauge', otherwise if not
                        provided, will be inferred from gauge data
  -e END, --end=END     Specify the end time in the format YYYYddmmHHMM, it is
                        mandatory only if type == 'gauge', otherwise if not
                        provided, will be inferred from gauge data
  -c CONFIG, --config=CONFIG
                        Path of the config file, the default will be
                        default_config.yml in the database module
  -g GAUGE, --gauge=GAUGE
                        Needed only if type == reference or radar, path
                        pattern (with wildcards) of the gauge data (from
                        database) to be used, default =
                        '/store/msrad/radar/radar_database/gauge/*.csv.gz',
                        IMPORTANT you have to put this statement into quotes
                        (due to wildcard)
                        
See ::doc::`db_options` to see how to define the configuration file.



**Example**

.. code-block:: console

    db_populate -t "reference" -g "/store/msrad/radar/radar_database/gauge/*.csv.gz" -o "/store/msrad/radar/radar_database/reference/"
    
