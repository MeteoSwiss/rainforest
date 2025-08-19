Database configuration
==========================================

When updating the database with new data either with the :doc:`command-line interface <interface>` or the :doc:`command-line tool <db_cmd>`, a configuration file is required.
This configuration file must be written in `YAML <https://fr.wikipedia.org/wiki/YAML/>`_, the default file is stored in *rainforest/database/default_config.yml* and has the following structure:

.. warning::
    It is highly recommended to use the standard default config file when updating the database, if you use different parameters, the database will not be consistent over time, so make sure you know what you are doing!
    
.. code-block:: yaml

    NO_DATA_FILL: -9999
    TMP_FOLDER: '/scratch/${USER}/temp/'
    GAUGE_RETRIEVAL:
        VARIABLES : ['tre200s0','prestas0','ure200s0','rre150z0','dkl010z0','fkl010z0']
        STATIONS : 'all'
        MIN_R_HOURLY : 0.1
        MAX_NB_SLURM_JOBS: 6

    REFERENCE_RETRIEVAL:
        PRODUCTS : ['RZC','CPCH','CPC.CV','BZC','MZC','MVRZC','MVCPCH']
        MV_METHOD: 'lucaskanade' # see https://pysteps.readthedocs.io/en/latest/generated/pysteps.motion.interface.get_method.html
        NEIGHBOURS_X : [-1,0,1] # not applied to CPC.CV
        NEIGHBOURS_Y : [-1,0,1] # not applied to CPC.CV
        MAX_NB_SLURM_JOBS: 20

    RADAR_RETRIEVAL:
        RADARS: ['A','D','L','W','P']
        RADAR_VARIABLES : ['ZH','ZV','ZH_VISIB','ZV_VISIB','ZDR','KDP','RHOHV','SW','RVEL','AH','ZH_CORR','ZV_CORR','ZDR_CORR','VISIB','NH','NV','HYDRO']
        COSMO_VARIABLES: ['T','U','V','W','P','QV'] # Only at center pixel
        OTHER_VARIABLES: ['HEIGHT','VPR','RADPRECIP'] # Only at center pixel
        AGGREGATION_METHODS: ['max','min','mean'] # value at max/min of kdp for kdp and max/min of zh for all other variables
        SWEEPS: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        NEIGHBOURS_X : [-1,0,1]
        NEIGHBOURS_Y : [-1,0,1]
        KDP_PARAMETERS:
            RMIN : 1000.
            RMAX : 50000.
            RCELL : 1000.
            ZMIN : 20

The parameters are the following, please note that environment variables must be indicated with ${name_of_variable}

-   **NO_DATA_FILL** : Used to indicate missing data in the tables
-   **TMP_FOLDER** : Temporary storage folder to use
-   **GAUGE_RETRIEVAL** : Options specific to the retrieval of station data
    
    -   **VARIABLES** : List of ground observation variables to retrieve (see Climap for a list of names)
    -   **STATIONS** :  List of stations for which to retrieve data as a list of `SMN abbreviations <https://www.meteosuisse.admin.ch/content/dam/meteoswiss/de/Mess-und-Prognosesysteme/doc/liste-der-geaenderten-stationsnamen_2017.pdf>`_, the keyword 'all', will use all stations present in the file *rainforest/common/constants/data_stations.csv*.
    -   **MIN_R_HOURLY**  : Minimum hourly rain total to consider the 10 min timesteps that fall within that hour. Default is 0.1 (one tip of the bucket).
    -   **MAX_NB_SLURM_JOBS** : Maximum number of SLURM jobs over which to share the processing. This will not affect the data.
    
-   **REFERENCE_RETRIEVAL** : Options specific to the retrieval of Cartesian MeteoSwiss products
    
    -   **PRODUCTS** : List of MeteoSwiss products to retrieve, the *MV* prefix indicates motion vectors derived from a given product
    -   **MV_METHOD** :  Name of the numerical method used to retrieve motion vectors, must be one of the following <https://pysteps.readthedocs.io/en/latest/generated/pysteps.motion.interface.get_method.html>`_
    -   **NEIGHBOURS_X**  : List of neighbours to get in the Swiss X direction (from south to north), 0 = location of the station, +1 = 1 pixel (1 km) to the north, -1, 1 pixel (1 km) to the south.
    -   **NEIGHBOURS_Y** : List of neighbours to get in the Swiss Y direction (from west to east), 0 = location of the station, +1 = 1 pixel (1 km) to the east, -1, 1 pixel (1 km) to the west.
-   **GAUGE_RETRIEVAL** : Options specific to the retrieval of station data
    
    -   **VARIABLES** : List of ground observation variables to retrieve (see Climap for a list of names)
    -   **STATIONS** :  List of stations for which to retrieve data as a list of `SMN abbreviations <https://www.meteosuisse.admin.ch/content/dam/meteoswiss/de/Mess-und-Prognosesysteme/doc/liste-der-geaenderten-stationsnamen_2017.pdf>`_, the keyword 'all', will use all stations present in the file *rainforest/common/constants/data_stations.csv*.
    -   **MIN_R_HOURLY**  : Minimum hourly rain total to consider the 10 min timesteps that fall within that hour. Default is 0.1 (one tip of the bucket).
    -   **MAX_NB_SLURM_JOBS** : Maximum number of SLURM jobs to use when updating the *gauge* table, this has no influence on the data
    
-   **RADAR_RETRIEVAL** : Options specific to the retrieval of polar radar products
    
    -   **RADARS** : List of radars for which to retrieve the data
    -   **RADAR_VARIABLES** : List of radar variables to retrieve, see :ref:`radtab` 
    -   **COSMO_VARIABLES**  : List of COSMO variables to retrieve, these are simplified GRIB names as obtained by using *Fieldextra*.
    -   **OTHER_VARIABLES** : List of other variables to retrieve, currently only ['HEIGHT','VPR','RADPRECIP'] are supported. These variables are available only at the location of the station (i.e. NX = NY = 0).
    -   **AGGREGATION_METHODS** : List of aggregation methods to use for *RADAR_VARIABLES*, 'mean' is the average in the Cartesian pixel, 'max' and 'min' are the values at the location of the max/min of ZH, except for KDP where the max and min of KDP over the Cartesian pixel is used instead.
    -   **NEIGHBOURS_X**  : List of neighbours to get in the Swiss X direction (from south to north), 0 = location of the station, +1 = 1 pixel (1 km) to the north, -1, 1 pixel (1 km) to the south.
    -   **NEIGHBOURS_Y** : List of neighbours to get in the Swiss Y direction (from west to east), 0 = location of the station, +1 = 1 pixel (1 km) to the east, -1, 1 pixel (1 km) to the west.
    -   **KDP_PARAMETERS** : set of parameters used in the computation of KDP using the moving least-square method. 

        -   **RMIN** : minimum range where to look for continuous precipitation, see `pyart code <https://github.com/meteoswiss-mdr/pyart/blob/master/pyart/correct/phase_proc.py>`_
        -   **RMAX** :  maximum range where to look for continuous precipitation, see `pyart code <https://github.com/meteoswiss-mdr/pyart/blob/master/pyart/correct/phase_proc.py>`_
        -   **ZMIN**  : minimum reflectivity to consider it a rain cell, see `pyart code <https://github.com/meteoswiss-mdr/pyart/blob/master/pyart/correct/phase_proc.py>`_
        -   **ZMAX**  : maximum reflectivity to consider it a rain cell, see `pyart code <https://github.com/meteoswiss-mdr/pyart/blob/master/pyart/correct/phase_proc.py>`_
        -   **RWIND** : size of the moving window in meters used in the PSIDP filtering and KDP estimation, see `pyart code <https://github.com/meteoswiss-mdr/pyart/blob/master/pyart/retrieve/kdp_proc.py>`_
    -   **SNR_THRESHOLD** : minimum SNR in dB below which the radar data is masked 
    -   **VISIB_CORR** : set of parameters for visibility correction

        -   **MIN_VISIB** : minimum visibility below which the data is masked
        -   **MAX_CORR** : maximum visibility correction for ZH (in linear)

    -   **MAX_SIMULTANEOUS_JOBS** : maximum number of SLURM jobs to run at the same time. The program will run in background and run additional jobs only if the current number of jobs is lower than this limit.
    -   **MAX_NB_SLURM_JOBS:** : Maximum number of SLURM jobs over which to share the processing. This will not affect the data.
    

