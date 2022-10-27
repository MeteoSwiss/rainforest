#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main class to update the RADAR/STATION database and run queries to retrieve
specific data


Note that I use spark because there is currently no way to use SQL queries
with dask
"""

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SQLContext, DataFrame

# This could benefit from some tweaks especially if the database becomes larger
conf = SparkConf()
conf.set("spark.sql.autoBroadcastJoinThreshold", 1024*1024*100)
conf.setAppName('Mnist_Spark_MLP').setMaster('local[8]')
conf.setAll([('spark.executor.memory', '8g'),
                    ('spark.executor.cores', '3'), 
                    ('spark.cores.max', '3'), 
                    ('spark.driver.memory','8g')])
conf.set("spark.sql.caseSensitive","true")

# Global imports
import glob
import yaml
import logging
logging.getLogger().setLevel(logging.INFO)
import os
import textwrap
import numpy as np
import subprocess
from datetime import datetime, timezone
import copy
import time
import fnmatch

# Local imports
from ..common import constants
from ..common.utils import chunks, timestamp_from_datestr
from ..common.utils import dict_flatten, read_df, envyaml

STATION_INFO = np.array(constants.METSTATIONS)

class TableDict(dict):
    """ This is an extension of the classic python dict that automatically
    calls createOrReplaceTempView once a table has been added to the dict """
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self[key].createOrReplaceTempView(key)
        
class DataFrameWithInfo(DataFrame):
     def __init__(self, name, df):

         super(self.__class__, self).__init__(df._jdf, df.sql_ctx)
         self.info = None
         self.name = name
     
     @property
     def info(self):
         if self.__info == None:
             cols = self.columns
             rows = self.count()
             times = self.select('timestamp').collect()
             t0 = datetime.utcfromtimestamp(np.min(times))
             t1 = datetime.utcfromtimestamp(np.max(times))
             
             self.__info = '''
             Table {:s} info
             ----------------
             Dimension: {:d} x {:d}
             Time interval: {:s} - {:s}
             Columns: {:s}
             '''.format(self.name, rows, len(cols), str(t0),str(t1),
             ','.join(cols))
         return self.__info
            
     @info.setter
     def info(self, value):
         self.__info = value

class Database(object):
    def __init__(self, config_file = None):
        """
        Creates a Database instance that can be used to load data, update
        new data, run queries, etc
        
        Parameters
        ----------
        config_file : str (optional)
            Path of the configuration file that you want to use, can also 
            be provided later and is needed only if you want to update the 
            database with new data
            
        """
        sparkContext = SparkContext(conf = conf)
        self.sqlContext = SQLContext(sparkContext)
        self.tables = TableDict()
        self.summaries = {}
        if config_file:
            self.config = envyaml(config_file)
            self.config_file = config_file
   
    @property
    def config_file(self):
        return self.__config_file

    @config_file.setter
    def config_file(self, config_file):
       self.config = envyaml(config_file)
       self.__config_file = config_file    
       
    def add_tables(self, filepaths_dic, get_summaries = False):
        """
        Reads a set of data contained in a folder as a Spark DataFrame and 
        adds them to the database instance
        
        Parameters
        ----------
        filepaths_dic : dict
            Dictionary where the keys are the name of the dataframes to add
            and the values are the wildcard patterns poiting to the files
            for example {'gauge': '/mainfolder/gauge/*.csv', 
                         'radar' : '/mainfolder/radar/*.csv',
                         'reference' : /mainfolder/reference/*.parquet'}
            will add the three tables 'gauge', 'radar' and 'reference' to the
            database       

        """
        for table in filepaths_dic:   
            pattern = filepaths_dic[table]

            self.tables[table] = DataFrameWithInfo(table, read_df(pattern,
                       dbsystem = 'spark', sqlContext = self.sqlContext))
           
            # Register as table
            self.tables[table].createOrReplaceTempView(table)
            
            # Below is experimental
            
#            # if get_summaries
#            if get_summaries:
#                summary_file = os.path.dirname(pattern)+'/.'+table
#                if os.path.exists(summary_file):
#                    self.summaries[table] = pd.read_csv(summary_file)
#                else:
#                    summary = self.tables[table].summary().toPandas()
#                    # Change timestamp to date and remove useless statistics
#                    dates = []
#                    if 'timestamp' in summary.columns:
#                        for i, stat in enumerate(summary['summary']):
#                            if stat not in ['min','max']:
#                                dates.append(np.nan)
#                            else:
#                                d = datetime.utcfromtimestamp(float(summary['timestamp'][i]))
#                                dates.append(d) 
#                        summary['date'] = dates
#                        summary = summary.drop('timestamp',1)
#                    if 'station' in summary.columns:
#                        summary = summary.drop('station',1)
#                        
#                    self.summaries[table] = summary
#                    self.summaries[table].to_csv(summary_file, index=False)
                
    def query(self, sql_query, to_memory = True, output_file = ''):
        """
        Performs an SQL query on the database and returns the result and if 
        wanted writes it to a file
        
        Parameters
        ----------
        sql_query : str
            Valid SQL query, all tables refered to in the query must be included
            in the tables attribute of the database (i.e. they must first
            be added with the add_tables command)
        to_ memory : bool (optional)
            If true will try to put the result into ram in the form of a pandas
            dataframe, if the predicted size of the query is larger than
            the parameter WARNING_RAM in common.constants this will be ignored
        output_file : str (optional)
            Full path of an output file where the query will be dumped into.
            Must end either with .csv, .gz.csv, or .parquet, this will 
            determine the output format
        
        Returns
        ----------
        If the result fits in memory, it returns a pandas DataFrame, otherwise
        a cached Spark DataFrame
        """
        
        sql_query = self._parse_query(sql_query)
        sqlDF = self.sqlContext.sql(sql_query)
        shape = _spark_shape(sqlDF)
        est_size = 10**-6 * (shape[0] * shape[1]) * 4
        
        if to_memory and est_size > constants.WARNING_RAM:
            logging.warning("""Query output is larger than maximum allowed size,
                         returning uncached version dataframe instead""")
            to_memory = False
        
        if to_memory:
            sqlDF = sqlDF.toPandas()
            
            if '.csv' in output_file:
                if '.gz' in output_file:
                    sqlDF.to_csv(output_file, compression = 'gzip', 
                                 index = False)
                else:
                    sqlDF.to_csv(output_file, 
                                 index = False)
                    
            elif 'parquet' in output_file:
                sqlDF.to_parquet(compression = 'GZIP')
        else:
            sqlDF = DataFrameWithInfo(sql_query, sqlDF)
            if '.csv' in output_file:
                if '.gz' in output_file:
                    sqlDF.write.csv(output_file, compression = 'GZIP',
                                    header = True)
                else:
                    sqlDF.write.csv(output_file, header = True)
            elif 'parquet' in output_file:
                sqlDF.write.parquet(output_file, compression = 'GZIP')
            
        return sqlDF
    
        
    def _parse_query(self, sql_query):
        '''
        Parses the query which could allow for custom keywords, 
        right now it just replaces UT with UNIX_TIMESTAMP
        '''
        # Search for Date) flags and replace with proper SQL keyword
        sql_query = sql_query.replace('UT(','UNIX_TIMESTAMP(')
        return sql_query
            
    def update_station_data(self, t0, t1, output_folder):
        '''
        update_station_data
            Populates the csv files that contain the point measurement data, 
            that serve as base to update the database. A different file will
            be created for every station. If the file is already present the
            new data will be 
            appended to the file.
        
        inputs:
            t0: start time in YYYYMMDD(HHMM) format (HHMM) is optional
            t1: end time in YYYYMMDD(HHMM) format (HHMM) is optional
            output_folder: where the files should be stored. If the directory
                is not empty, the new data will be merged with existing files
                if relevant
        
        '''
        
        if not output_folder.endswith(os.path.sep):
            output_folder += os.path.sep
            
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        try:
            self.config
        except:
            raise ValueError('Need to provide a config file to update data')
            
        if 'GAUGE_RETRIEVAL' in self.config:
            config_g = self.config['GAUGE_RETRIEVAL']
        else:
            raise ValueError("""Make sure you have a "GAUGE_RETRIEVAL" section in 
                          your config file!""")
        
        
        tmp_folder = self.config['TMP_FOLDER']
        
        if config_g['STATIONS'] == 'all':
            stations = STATION_INFO[:,1]
        elif config_g['STATIONS'] == 'all_smn':
            stations = STATION_INFO[STATION_INFO[:,6] == 'SwissMetNet',1]
        elif config_g['STATIONS'] == 'all_ps':
            stations = STATION_INFO[STATION_INFO[:,6] == 'PrecipStation',1]
        else:
            stations = config_g['STATIONS'] 
            if type(stations) != list:
                stations = [stations]

        # Check if metadata file already present in folder
                # Check existence of previous data
        try:
            # Try to read old data
            current_tab = read_df(output_folder + '*.csv*', 
                                       dbsystem = 'spark',
                                       sqlContext = self.sqlContext)
            old_data_ok = True # valid data present
        except:
            old_data_ok = False
            pass
                
        # Check existence of config file
        mdata_path = output_folder + '/.mdata.yml'
        try:
            old_config = envyaml(mdata_path)['GAUGE_RETRIEVAL']
        except:
            old_config = None
            pass
        
        overwrite = 1 # default

        if old_config and old_data_ok:
            if _compare_config(old_config, self.config, 
                       ['GAUGE_RETRIEVAL','NO_DATA_FILL']):
                # Old data same format as new, don't overwrite
                overwrite = 0
            if old_config != self.config:
                # Special case
                tsteps_proc = np.array(current_tab.select('timestamp').collect(), 
                                       dtype=int)
                tstamp_start_old = np.min(tsteps_proc)
                tstamp_end_old = np.max(tsteps_proc)
                
                tstamp_start = int(timestamp_from_datestr(t0))
                tstamp_end = int(timestamp_from_datestr(t1))
            
            
                if (tstamp_start > tstamp_start_old or
                    tstamp_end < tstamp_end_old):
                    warning = """
                    IMPORTANT: A previous set of tables was found in the indicated output folder
                    corresponding to a different configuration file. If you continue, the old data
                    will be replaced by the newly generated data, if they have the same timestamps.
                    HOWEVER since the new data does not temporally cover the full extent of the old data,
                    old and new data will coexist in the folder, which is ABSOLUTELY not recommended.
                    If you are not sure what to do, either 
                    (1) delete the old data
                    (2) change the current config file to match the config file of the old data 
                    (which is stored in the file .mdata.yaml in the specified output folder)
                    (3) Rerun retrieval of station data to overlap the time period
                    covered by the old data ({:s} - {:s}) and rerun the radar retrieval
                    
                    Press enter to continue, q to quit ...
                    """.format(str(datetime.utcfromtimestamp(tstamp_start_old)),
                               str(datetime.utcfromtimestamp(tstamp_end_old)))
                    userinput = input(textwrap.dedent(warning))
                    if userinput == 'q':
                        raise KeyboardInterrupt()                                     
        
        # Write metadata file
        mdata = copy.deepcopy(config_g)
        yaml.dump(mdata, open(mdata_path,'w'))

        # If timestamp is datetime format, convert to string
        if isinstance(t0, datetime):
            t0 = t0.strftime("%Y%m%d%H%M")
        if isinstance(t1, datetime):        
            t1 = t1.strftime("%Y%m%d%H%M") 
        
        # Split all stations in subsets
        max_nb_jobs = config_g['MAX_NB_SLURM_JOBS']
        stations_sub = chunks(stations, max_nb_jobs)
        
        # Get current folder
        cwd = os.path.dirname(os.path.realpath(__file__))
        
        for i, stations in enumerate(stations_sub):
            fname = tmp_folder + '/getdata_station_{:d}.job'.format(i)
            file = open(fname,'w')
            logging.info('Writing task file {:s}'.format(fname))
            file.write(constants.SLURM_HEADER_R)
            file.write('Rscript {:s}/retrieve_dwh_data.r "{:s}" "{:s}" {:f} "{:s}" "{:s}" {:s} {:d} {:d}'.format(
                       cwd,
                       t0,
                       t1,
                       config_g['MIN_R_HOURLY'],
                       ','.join(stations),
                       ','.join(config_g['VARIABLES']),
                       output_folder,
                       self.config['NO_DATA_FILL'],
                       overwrite))
            file.close()
            logging.info('Submitting job {:d}'.format(i))
            subprocess.call('sbatch {:s}'.format(fname), shell = True)
        logging.info("""All jobs have been submitted, please wait a few hours
                     for completion...""")
     
    def update_reference_data(self, gauge_table_name,  output_folder, 
                              t0 = None, t1 = None):
        '''
        Updates the reference product table using timesteps from the gauge table
        
        Inputs:
            gauge_table_name: str
                name of the gauge table, must be included in the tables of 
                the database, i.e. you must first add it with load_tables(..)
            output_folder: str
                directory where to store the computed radar tables
            t0: start time in YYYYMMDD(HHMM) (optional)
                starting time of the retrieval, by default all timesteps 
                that are in the gauge table will be used
            t1: end time in YYYYMMDD(HHMM) (optional)
                ending time of the retrieval, by default all timesteps 
                that are in the gauge table will be used
        '''
        if not output_folder.endswith(os.path.sep):
            output_folder += os.path.sep
            
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        try:
            self.config
        except:
            raise ValueError('Need to provide a config file to update data')
          
        if gauge_table_name not in self.tables.keys():
            raise ValueError("""No table with name {:s} was found in the
                             loaded tables, make sure to add that table
                             with load_tables first""".format(gauge_table_name))
        
        if 'REFERENCE_RETRIEVAL' in self.config:
            config_r = self.config['REFERENCE_RETRIEVAL']
        else:
            raise ValueError("""Make sure you have a "REFERENCE_RETRIEVAL" section in 
                          your config file!""")
        
                # Check existence of previous data
        try:
            # Try to read old data
            current_tab = read_df(output_folder + '*.parquet',
                                       dbsystem = 'spark',
                                       sqlContext = self.sqlContext)
            old_data_ok = True # valid data present
        except:
            old_data_ok = False
            pass
    
     
        logging.info('Finding unique timesteps and corresponding stations')
        tab = self.tables[gauge_table_name].select(['STATION',
                        'TIMESTAMP']).toPandas()
    
        if t0 != None and t1 != None and t1 > t0:
            logging.info('Limiting myself to time period {:s} - {:s}'.format(
                    str(t0), str(t1)))
            if not isinstance(t0, datetime):
                tstamp_start = int(timestamp_from_datestr(t0))
                tstamp_end = int(timestamp_from_datestr(t1))
            else:
                t0 = t0.replace(tzinfo=timezone.utc)
                t1 = t1.replace(tzinfo=timezone.utc)
                tstamp_start = int(t0.timestamp())
                tstamp_end = int(t1.timestamp())
                
            tab = tab.loc[(tab['TIMESTAMP'] > tstamp_start) 
                            & (tab['TIMESTAMP'] <= tstamp_end)]

        # Check existence of config file
        mdata_path = output_folder + '/.mdata.yml'
        try:
            old_config = envyaml(mdata_path)
        except:
            old_config = None
            pass
      
        overwrite = 1 # default
        if old_config and old_data_ok:
            if _compare_config(old_config, self.config, 
                       ['GAUGE_RETRIEVAL','REFERENCE_RETRIEVAL','NO_DATA_FILL']):
                # Old data same format as new, don't overwrite
                overwrite = 0
            else:
                # Special case
       
                tsteps_proc = current_tab.select('TIMESTAMP').collect()
                tstamp_start_old = np.min(tsteps_proc)
                tstamp_end_old = np.max(tsteps_proc)
                
                tstamp_start = int(np.min(tab['TIMESTAMP']))
                tstamp_end = int(np.max(tab['TIMESTAMP']))
         
                
                if (tstamp_start > tstamp_start_old or
                    tstamp_end < tstamp_end_old):
                    warning = """
                    IMPORTANT: A previous set of tables was found in the indicated output folder
                    corresponding to a different configuration file. If you continue, the old data
                    will be replaced by the newly generated data, if they have the same timestamps.
                    HOWEVER since the new data does not temporally cover the full extent of the old data,
                    old and new data will coexist in the folder, which is ABSOLUTELY not recommended.
                    If you are not sure what to do, either 
                    (1) delete the old data
                    (2) change the current config file to match the config file of the old data 
                    (which is stored in the file .mdata.yaml in the specified output folder)
                    (3) Rerun retrieval of station data to overlap the time period
                    covered by the old data ({:s} - {:s}) and rerun the radar retrieval
                    
                    Press enter to continue, q to quit ...
                    """.format(str(datetime.utcfromtimestamp(tstamp_start_old)),
                               str(datetime.utcfromtimestamp(tstamp_end_old)))
                    userinput = input(textwrap.dedent(warning))
                    if userinput == 'q':
                        raise KeyboardInterrupt()
    
        # Write metadata file
        mdata = copy.deepcopy(self.config)
        yaml.dump(mdata, open(mdata_path,'w'))

        unique_times, idx = np.unique(tab['TIMESTAMP'], return_inverse = True)
        
        if not len(unique_times):
            msg = '''All timesteps are already present in the already computed tables in the indicated output folder!'''
            logging.error(textwrap.dedent(msg))
            logging.error('Stopping now...')
            return
            
        all_stations = tab['STATION']
        
        # Split tasks and write taskfiles
        logging.info('Writing task files, this can take a long time')
        num_jobs = config_r['MAX_NB_SLURM_JOBS']
        tmp_folder = self.config['TMP_FOLDER']
        
        # Jobs are split by days, a single day is never split over several jobs
        # because the created output files are day based
        
        ttuples = [datetime.utcfromtimestamp(float(t)).timetuple()
            for t in unique_times]

        days = [[str(t.tm_year) + str(t.tm_yday)] for t in ttuples]
        days_to_process = np.unique(days, axis = 0)
        days_to_process = list(days_to_process)
        
        if not overwrite:
            msg = '''A previous set of tables corresponding to the same config file was found, only new timestamps will be added'''
            logging.warning(textwrap.dedent(msg))
            # Find which days have already been processed and remove them

            files = glob.glob(output_folder + '*.parquet')
            for f in files:
                
                f = os.path.splitext(os.path.basename(f))[0]
                dt = datetime.strptime(os.path.basename(f),'%Y%m%d')
                tt = dt.timetuple()
                current_day = str(tt.tm_year) + str(tt.tm_yday)
                if current_day in days_to_process:
                    logging.warning('Day {:s} was already computed, ignoring it...'.format(f))
                    days_to_process.remove(current_day)
                    
        days_per_job = max([1,int(np.round(len(days_to_process)/num_jobs))])
        
        day_counter = 0
        current_job = 0
        current_day = days[0]
        
        # Create task files
        task_files = []
        name_file = tmp_folder + 'task_file_reference_{:d}'.format(current_job)
        task_files.append(name_file)
        ftask = open(name_file,'w')
        
        for i in range(len(unique_times)):
            
            if days[i] not in days_to_process:
                continue
            if days[i] != current_day:
                
                day_counter += 1
                current_day = days[i]
                
            if day_counter == days_per_job:
                # New job
                current_job += 1
                ftask.close()
                
                name_file = tmp_folder + 'task_file_reference_{:d}'.format(current_job)
                logging.info('Writing task file {:s}'.format(name_file))
                task_files.append(name_file)
                # Open new task file
                ftask = open(name_file,'w')
                # Reset counters
                day_counter = 0
            ftask.write('{:d},{:s} \n'.format(int(unique_times[i]),
                    ','.join(all_stations[idx == i])))    

        ftask.close()

        # Get current folder
        cwd = os.path.dirname(os.path.realpath(__file__))
        # Create slurm files
        job_files = []
        nfperjob = config_r['SLURM_JOBS_PER_FILE']
        jobmax = config_r['MAX_SIMULTANEOUS_JOBS']
        tf = tmp_folder + 'task_file_reference_${SLURM_ARRAY_TASK_ID}'
        
        slurm_header = '#SBATCH --output="db_ref_%A_%a.out"\n'+ \
                       '#SBATCH --error="db_ref_%A_%a.err"\n' + \
                       '#SBATCH --job-name=DB_REF\n'

        slurm_python_setup = "source /scratch/rgugerli/miniconda3/etc/profile.d/conda.sh\n" + \
                            "conda activate {}\n\n".format(self.config['CONDA_ENV_NAME']) + \
                            "export RAINFOREST_DATAPATH=/store/msrad/radar/rainforest/rainforest_data/\n\n"

        # If only one task file is created:
        if len(task_files) == 1:
            fname = tmp_folder + '/getdata_reference_0.job'
            file = open(fname,'w')
            file.write(constants.SLURM_HEADER_PY)
            file.write(slurm_header)
            file.write('#SBATCH --array=0\n\n')
            # The following two lines are necessary to use the right environment
            file.write(slurm_python_setup)
            file.write('python {:s}/retrieve_reference_data.py -c {:s} -t {:s} -o {:s} \n'.format(
                       cwd,
                       self.config_file,
                       tf,
                       output_folder))
            file.close()
            job_files= [fname]
        # If more than one task file is created loop through them and created grouped batch files          
        else:
            i=0
            while i < (len(task_files)-1):
                if (i+nfperjob) > (len(task_files)-1):
                    iend = (len(task_files)-1)
                else:
                    iend = i+nfperjob
                fname = tmp_folder + '/getdata_reference_{:d}_{:d}.job'.format(i,iend)
                file = open(fname,'w')
                file.write(constants.SLURM_HEADER_PY)
                file.write(slurm_header)
                file.write('#SBATCH --array={:d}-{:d}%{:d}\n\n'.format(i,iend,jobmax))
                file.write(slurm_python_setup)
                file.write('python {:s}/retrieve_reference_data.py -c {:s} -t {:s} -o {:s} \n'.format(
                        cwd,
                        self.config_file,
                        tf,
                        output_folder))
                file.close()
                # Set counter to next file that is not included (+1)
                i = i+1+nfperjob
                job_files.append(fname)

        for fn in job_files:
            logging.info('Submitting job {}'.format(fn))
            #subprocess.call('sbatch {:s}'.format(fn), shell = True)
              
        
    def update_radar_data(self, gauge_table_name,  output_folder,
                          t0 = None, t1 = None):
        '''
        Updates the radar table using timesteps from the gauge table
        
        Inputs:
            gauge_table_name: str
                name of the gauge table, must be included in the tables of 
                the database, i.e. you must first add it with load_tables(..)
            output_folder: str
                directory where to store the computed radar tables
            t0: start time in YYYYMMDD(HHMM) (optional)
                starting time of the retrieval, by default all timesteps 
                that are in the gauge table will be used
            t1: end time in YYYYMMDD(HHMM) (optional)
                ending time of the retrieval, by default all timesteps 
                that are in the gauge table will be used
        '''
        
        if not output_folder.endswith(os.path.sep):
            output_folder += os.path.sep
            
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        try:
            self.config
        except:
            raise ValueError('Need to provide a config file to update data')
            
        if gauge_table_name not in self.tables.keys():
            raise ValueError("""No table with name {:s} was found in the
                             loaded tables, make sure to add that table
                             with load_tables first""".format(gauge_table_name))
        
        if 'RADAR_RETRIEVAL' in self.config:
            config_r = self.config['RADAR_RETRIEVAL']
        else:
            raise ValueError("""Make sure you have a "RADAR_RETRIEVAL" section in 
                          your config file!""")
        
        logging.info('Finding unique timesteps and corresponding stations')
        tab = self.tables[gauge_table_name].select(['STATION',
                        'TIMESTAMP']).toPandas()
        
        if t0 != None and t1 != None and t1 > t0:
            logging.info('Limiting myself to time period {:s} - {:s}'.format(
                    str(t0), str(t1)))
            if not isinstance(t0, datetime):
                tstamp_start = int(timestamp_from_datestr(t0))
                tstamp_end = int(timestamp_from_datestr(t1))
            else:
                t0 = t0.replace(tzinfo=timezone.utc)
                t1 = t1.replace(tzinfo=timezone.utc)
                tstamp_start = int(t0.timestamp())
                tstamp_end = int(t1.timestamp())
                
            tab = tab.loc[(tab['TIMESTAMP'] > tstamp_start) 
                            & (tab['TIMESTAMP'] <= tstamp_end)]
            
        # Check existence of previous data
        try:
            # Try to read old data
            current_tab = read_df(output_folder + '*.parquet',
                                       dbsystem = 'spark',
                                       sqlContext = self.sqlContext)
            old_data_ok = True # valid data present
        except:
            old_data_ok = False
            pass
                
        # Check existence of config file
        mdata_path = output_folder + '/.mdata.yml'
        try:
            old_config = envyaml(mdata_path)
        except:
            old_config = None
            pass
        
        overwrite = 1 # default
        if old_config and old_data_ok:
            if _compare_config(old_config, self.config, 
                               ['GAUGE_RETRIEVAL','RADAR_RETRIEVAL',
                               'NO_DATA_FILL']):
                # Old data same format as new, don't overwrite
                overwrite = 0
            else:
                # Special case
                tsteps_proc = current_tab.select('TIMESTAMP').collect()
                tstamp_start_old = np.min(tsteps_proc)
                tstamp_end_old = np.max(tsteps_proc)
                
                tstamp_start = int(np.min(tab['TIMESTAMP']))
                tstamp_end = int(np.max(tab['TIMESTAMP']))
                    
                if (tstamp_start > tstamp_start_old or
                    tstamp_end < tstamp_end_old):
                    warning = """
                    IMPORTANT: A previous set of tables was found in the indicated output folder
                    corresponding to a different configuration file. If you continue, the old data
                    will be replaced by the newly generated data, if they have the same timestamps.
                    HOWEVER since the new data does not temporally cover the full extent of the old data,
                    old and new data will coexist in the folder, which is ABSOLUTELY not recommended.
                    If you are not sure what to do, either 
                    (1) delete the old data
                    (2) change the current config file to match the config file of the old data 
                    (which is stored in the file .mdata.yaml in the specified output folder)
                    (3) Rerun retrieval of station data to overlap the time period
                    covered by the old data ({:s} - {:s}) and rerun the radar retrieval
                    
                    Press enter to continue, q to quit ...
                    """.format(str(datetime.utcfromtimestamp(tstamp_start_old)),
                               str(datetime.utcfromtimestamp(tstamp_end_old)))
                    userinput = input(textwrap.dedent(warning))
                    if userinput == 'q':
                        raise KeyboardInterrupt()
                        
        # Write metadata file
        mdata = copy.deepcopy(self.config)
        yaml.dump(mdata, open(mdata_path,'w'))
        
        unique_times, idx = np.unique(tab['TIMESTAMP'], return_inverse = True)
        all_stations = tab['STATION']
        
        if not len(unique_times):
            msg = '''All timesteps are already present in the already computed tables in the indicated output folder!'''
            logging.error(textwrap.dedent(msg))
            logging.error('Stopping now...')
            return
        
        # Split tasks and write taskfiles
        num_jobs = config_r['MAX_NB_SLURM_JOBS']
        tmp_folder = self.config['TMP_FOLDER']
        
        # Jobs are split by days, a single day is never split over several jobs
        # because the created output files are day based
        
        ttuples = [datetime.utcfromtimestamp(float(t)).timetuple()
            for t in unique_times]

        days = [[str(t.tm_year) + str(t.tm_yday)] for t in ttuples]
        days_to_process = np.unique(days, axis = 0)
        days_to_process = list(days_to_process)

        if not overwrite:
            msg = '''A previous set of tables corresponding to the same config file was found, only new timestamps will be added'''
            logging.warning(textwrap.dedent(msg))
            # Find which days have already been processed and remove them

            files = glob.glob(output_folder + '*.parquet')
            for f in files:
                
                f = os.path.splitext(os.path.basename(f))[0]
                dt = datetime.strptime(os.path.basename(f),'%Y%m%d')
                tt = dt.timetuple()
                current_day = str(tt.tm_year) + str(tt.tm_yday)
                
                if current_day in days_to_process:
                    logging.warning('Day {:s} was already computed, ignoring it...'.format(f))
                    days_to_process.remove(current_day)
    
        days_per_job = max([1,int(np.round(len(days_to_process)/num_jobs))])

        day_counter = 0
        current_job = 0
        current_day = days[0]
        
        logging.info('Writing task files')
        # Create task files
        task_files = []
        name_file = tmp_folder + 'task_file_radar_{:d}'.format(current_job)
        task_files.append(name_file)
        ftask = open(name_file,'w')

        for i in range(len(unique_times)):
            if days[i] not in days_to_process:
                continue
            
            if days[i] != current_day:
                
                day_counter += 1
                current_day = days[i]
                
            if day_counter == days_per_job:
                # New job
                current_job += 1
                ftask.close()
                
                name_file = tmp_folder + 'task_file_radar_{:d}'.format(current_job)
                logging.info('Writing task file {:s}'.format(name_file))
                
                task_files.append(name_file)
                ftask = open(name_file,'w')
                # Reset counters
                day_counter = 0
            ftask.write('{:d},{:s} \n'.format(int(unique_times[i]),
                    ','.join(all_stations[idx == i])))    

        ftask.close()

        # Get current folder
        cwd = os.path.dirname(os.path.realpath(__file__))
        
        # Create slurm files
        job_files = []
        nfperjob = config_r['SLURM_JOBS_PER_FILE']
        jobmax = config_r['MAX_SIMULTANEOUS_JOBS']
        tf = tmp_folder + task_files[0].split('/')[-1][0:15]+'_${SLURM_ARRAY_TASK_ID}'
        
        slurm_header = '#SBATCH --output="db_radar_%A_%a.out"\n'+ \
                       '#SBATCH --error="db_radar_%A_%a.err"\n' + \
                       '#SBATCH --job-name=DB_RADAR\n'

        slurm_python_setup = "source /scratch/rgugerli/miniconda3/etc/profile.d/conda.sh\n" + \
                            "conda activate {}\n\n".format(self.config['CONDA_ENV_NAME']) + \
                            "export RAINFOREST_DATAPATH=/store/msrad/radar/rainforest/rainforest_data/\n\n"

        # If only one task file is created:
        if len(task_files) == 1:
            fname = tmp_folder + '/getdata_radar_0.job'
            file = open(fname,'w')
            file.write(constants.SLURM_HEADER_PY)
            file.write(slurm_header)
            file.write('#SBATCH --array=0\n\n')
            file.write(slurm_python_setup)
            file.write('python {:s}/retrieve_radar_data.py -c {:s} -t {:s} -o {:s} \n'.format(
                       cwd,
                       self.config_file,
                       tf,
                       output_folder))
            file.close()
            job_files= [fname]
        # If more than one task file is created loop through them and created grouped batch files          
        else:
            i=0
            while i < (len(task_files)-1):
                if (i+nfperjob) > (len(task_files)-1):
                    iend = (len(task_files)-1)
                else:
                    iend = i+nfperjob
                fname = tmp_folder + '/getdata_radar_{:d}_{:d}.job'.format(i,iend)
                file = open(fname,'w')
                file.write(constants.SLURM_HEADER_PY)
                file.write(slurm_header)
                file.write('#SBATCH --array={:d}-{:d}%{:d}\n\n'.format(i,iend,jobmax))
                file.write(slurm_python_setup)
                file.write('python {:s}/retrieve_radar_data.py -c {:s} -t {:s} -o {:s} \n'.format(
                        cwd,
                        self.config_file,
                        tf,
                        output_folder))
                file.close()
                # Set counter to next file that is not included (+1)
                i = i+1+nfperjob
                job_files.append(fname)

        for fn in job_files:
            logging.info('Submitting job {}'.format(fn))
            #subprocess.call('sbatch {:s}'.format(fn), shell = True)

        
def _compare_config(config1, config2, keys = None):
    """
    Compares the configuration of two data tables, by checking only the keys
    that affect the data (i.e. the radar processing, the choice of samples)
    
    Parameters
    ----------
    config1 : dict
        configuration dictionary 1
    config2 : dict
        configuration dictionary 2
    keys : which dict keys to check, by default all are checked
    
    Returns
    -------
    True if two configurations are equivalent, False otherwise
    """
    if keys == None:
        keys = list(config1.keys())
    # Returns True if the config files are the same, in terms of data content
    # Things like, MAX_NB_SLURM_JOBS or MAX_SIMULTANEOUS_JOBS don't matter
    keys_no_data = ['MAX_NB_SLURM_JOBS','TMP_FOLDER','MAX_SIMULTANEOUS_JOBS']
    c1 = dict_flatten(config1)
    c2 = dict_flatten(config2)
    
    try:
        for k in c1.keys():
            if k not in keys:
                continue
            notimportant = any([knd in k for knd in keys_no_data])
            if not notimportant:
                if c1[k] != c2[k]:
                    return False
        for k in c2.keys():
            if k not in keys:
                continue
            notimportant = any([knd in k for knd in keys_no_data])
            if not notimportant:
                if c1[k] != c2[k]:
                    return False        
        return True
    except:
        return False

def _n_running_jobs(user = '$USER', job_name = 'getdata*'):
    """
    Gets the number of jobs currently running on CSCS
    
    Parameters
    ----------
    user : str
        the user on the CSCS servers
    job_name : str
        name of the job, UNIX style wildcards are supported
   
    Returns
    -------
    Number of jobs as an integer
    """

    out = subprocess.check_output('squeue -u {:s}'.format(user),
                                  shell=True)
    
    out = out.decode('utf-8').split('\n')
    
    if len(out) == 2:
        return 0
    
    count = 0
    for l in out[1:-1]:
        l = l.split()
        if len(fnmatch.filter([l[2]],job_name)):
            count += 1
    return count

def _spark_shape(df):
    return (df.count(),len(df.columns))
