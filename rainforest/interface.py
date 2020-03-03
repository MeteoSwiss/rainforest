#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command line interface to the database


"""


# Global imports
import datetime
import os
from pathlib import Path
from prompt_toolkit.shortcuts import prompt
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit import print_formatted_text
import json
import yaml
import subprocess

# Local imports
from rainforest.database.database import Database, DataFrameWithInfo
from rainforest.common import constants

print = print_formatted_text


def prompt_check(promptext, check, **kwargs):
    success = False
    while not success:
        inpt = prompt(promptext, **kwargs)
        if check == None:
            success = True
        else:
            success = check_input(inpt, check)
    return inpt

def check_input(inpt, check):
    try:
        if type(check) == list:
            assert(any([check_input(inpt, c) for c in check]))
        else:
            if '%Y' in check: # datetime input
                datetime.datetime.strptime(inpt, check)
            elif check == 'json':
                json.loads(inpt)
            elif check == 'yaml':
                yaml.load(open(inpt,'r'), 
                               Loader = yaml.FullLoader)
            elif check == 'pathexists':
                assert(os.path.exists(inpt))
            elif check == 'makedirs':
                if not os.path.exists(inpt):
                    os.makedirs(inpt)
            else:
                assert(inpt == check)
            
        success = True
    except:
        success = False
    return success

def main(): # protect
    RADAR_DB_PATH = '/store/msrad/radar/radar_database/'
    
    style_info = Style.from_dict({
        # User input (default text).
        'command':          '#54B5EE',
        '': '#DCDCDC',
    })
    
    style = Style.from_dict({
        # User input (default text).
        '':          '#ff0066',
    
        # Prompt.
        'username': '#54B5EE',
    })
    
    
    style_prompt = Style.from_dict({
        # User input (default text).
        '':          '#DCDCDC',
    })
    
    
    style_warning = Style.from_dict({
        # User input (default text).
        '':          '#FF4500',
    })
    
    style_ok = Style.from_dict({
        # User input (default text).
        '':          '#00FF7F',
    })
    
    
    title1 = """-----------------------------------------
QPE database python interface
Daniel Wolfensberger, LTE-MeteoSwiss, 2019
------------------------------------------"""
    
    title2 = """-----------------------------------------
Database operations menu
------------------------------------------"""
    
    title3 = """-----------------------------------------
QPE menu
------------------------------------------"""
    
    
    dbase = Database()
    
    current_menu = 'main'
    
    info = {}
    info['main'] = FormattedText([
            ('class:command','db'),
            ('',': Enter database submenu (e.g. data queries, addition of new data) \n'),
            ('class:command','qpe'),
            ('',': Enter qpe submenu (e.g. compute qpe, generate maps) \n'),   
            ('class:command','e'),
            ('',': exit program \n')])
    
    info['db'] = FormattedText([
            ('class:command','load_cscs'),
            ('',': load all tables available on CSCS \n'),    
            ('class:command','load'),
            ('',': load one or several new tables from files \n'),
            ('class:command','query'),
            ('',': run a new query \n'),
            ('class:command','populate'),
            ('',': populate the database with new data \n'),
            ('class:command','display <name_of_table> n'),
            ('',': displays n rows of a loaded table \n')])
    
    info['qpe'] = FormattedText([
            ('class:command','compute'),
            ('',': compute the QPE estimate for a given time period \n'),
            ('class:command','plot'),
            ('',': plot a set of QPE estimates of a given time period \n')])
    
    print(title1, style=style)
    
    code = None
    current_query = None
    while(code != 'e'):
        try:
            code = prompt('Enter command (i for info), use ctrl+c to return to main menu anytime ',
                          style = style_prompt)
            ########
            # Info
            ########
            if code == 'i':
                print(info[current_menu], style=style_info)
    
            if current_menu == 'main':
                if code == 'db':
                    print(title2, style=style)
                    current_menu = 'db'
                elif code == 'qpe':
                    print(title3, style=style)
                    current_menu = 'qpe'
    
            ########
            # Populate
            ########
            if current_menu == 'db':
                if 'display' in code:
                    nametab = code.split(' ')[1]
                    try:
                        nrows = int(code.split(' ')[2])
                    except:
                        print('Invalid number of rows, using 10',
                              style = style_warning)
                        nrows = 10
                        
                    if nametab not in dbase.tables.keys():
                        print('Table name {:s} is not in loaded table names: {:s}'.format(
                            nametab, ','.join(list(dbase.tables.keys()))), 
                              style = style_warning)
                    else:
                        dbase.tables[nametab].show(nrows)

                elif code == 'populate':
    
                    n = prompt_check('With which type of data would you like to populate the database: "gauge", "radar" or "reference"? ',
                                     ['gauge','radar','reference'])
                    
                    if n == 'gauge':
                        txt = 'Indicate start time of the data you want to add (format YYYYMMDDHHMM, HHMM is optional) '
                        t0 = prompt_check(txt, ['%Y%m%d','%Y%m%d%H%M'])
                    else:
                        txt = 'Indicate start time of the data you want to add (format YYYYmmddHMM, HHMM is optional), leave empty to select automatically from gauge data: '
                        t0 = prompt_check(txt,  ['%Y%m%d','%Y%m%d%H%M',''])
                        
                    if n == 'gauge':
                        txt = 'Indicate end time of the data you want to add (format YYYYMMDDHHMM, HHMM is optional) '
                        t1 = prompt_check(txt, ['%Y%m%d','%Y%m%d%H%M'])
                    else:
                        txt = 'Indicate end time of the data you want to add (format YYYYmmddHMM, HHMM is optional), leave empty to select automatically from gauge data: '
                        t1 = prompt_check(txt,  ['%Y%m%d','%Y%m%d%H%M',''])
                    
                    if n != 'gauge':   
                        txt = """Select the gauge tables that will be used as a reference to select timesteps, indicate either the filepaths or the name of a table that has previously been added with the load table instruction: """
                        g = prompt_check(txt, list(dbase.tables.keys()))
               
    
                    o = prompt_check('Enter the location where the generated files will be stored: ',
                                     check = 'makedirs',
                                     default = str(Path(RADAR_DB_PATH, n)) + os.sep)
                 
                    script_path = os.path.dirname(os.path.realpath(__file__)) 
                    default_config_path = Path(script_path, 'database', 
                                                   'default_config.yml')
                   
                    c = prompt_check('Enter location of the configuration file (in yml format): ',
                                check = 'yaml',
                                default = str(default_config_path))
                    
                    print('You want to update the database with the following parameters...')
                    print('Data type: '  + n)
                    print('Starting time: ' + str(t0))
                    print('End time: ' + str(t1))
                    if n != 'gauge':
                        print('Gauge reference: ' + g )
                    print('Output folder: ' + o)
                    print('Config file: '+c)
                                    
                    ok = prompt('Do you want to start y/n: ')
                    if ok == 'y':
                        dbase.config_file = c
                        if n == 'gauge':
                            dbase.update_station_data(t0, t1, o)
                        elif n == 'reference':
                            dbase.update_reference_data(g, o, t0, t1)
                        elif n == 'radar':
                            dbase.update_radar_data(g, o, t0, t1)
                        
                ########
                # Load
                ########
                elif code == 'load_cscs':
                    dic = {'gauge': RADAR_DB_PATH + 'gauge/*.csv.gz',
                           'radar' : RADAR_DB_PATH + 'radar/*.parquet',
                           'reference': RADAR_DB_PATH + 'reference/*.parquet'}
                    try:
                        dbase.add_tables(dic)
                        print('The CSCS tables, "radar" "reference" and "gauge" were successfully added', style = style_ok)
                    except Exception as e:
                        print('Could not CSCS add table!', style = style_warning)
                        print(e, style = style_warning)
                        
                elif code == 'load':
                    n = prompt('Enter name of table(s) (you choose), use comma to separate multiple entries: ', default = 'radar')
                    if n == 'gauge':
                        default_suf = '*.csv.gz'
                    else:
                        default_suf = '*.parquet'
                    d = prompt('Enter filepaths (ex. /mydir/*.csv) where the table(s) are stored, use comma to separate multiple entries: ', 
                               default = str(Path(RADAR_DB_PATH, n, default_suf)))
            
                    try:
                        dic = {}
                        for nn,dd in zip(n.split(','), d.split(',')):
                            dic[nn] = dd
                        dbase.add_tables(dic)
                        print('The table was successfully added', style = style_ok)
                    except Exception as e:
                        print('Could not add table!', style = style_warning)
                        print(e, style = style_warning)
                 
                elif code == 'query':
                    q =  prompt('Enter your SQL query: ')
                    try:
                        current_query = dbase.query(q, to_memory = False)
                    except Exception as e:
                        current_query = None
                        print(e, style = style_warning)
                    
                    if current_query != None:
                        txt = 'Enter a filename if you want to save query (.csv, .csv.gz or .parquet), leave empty to pass: '
                        f =  prompt(txt)
                        if f != '':
                            if '.csv' in f:
                                if '.gz' in f:
                                    current_query.toPandas().to_csv(f, compression = 'gzip', 
                                                 index = False)
                                else:
                                    current_query.toPandas().to_csv(f, 
                                                 index = False)
                            elif 'parquet' in f:
                                current_query.toPandas().to_parquet(compression = 'GZIP')
                                
                        txt = 'Enter name if you want to add query as a table to the dataset, leave empty to pass: '
                        a =  prompt(txt)
                        if a != '':
                             dbase.tables[a] = DataFrameWithInfo(a, current_query)
                    
            elif current_menu == 'qpe':
                if code == 'compute':
                    ########
                    # Compute new qpe
                    ########
                    txt = 'Indicate start time of the data you want to add (format YYYYMMDDHHMM, HHMM is optional) '
                    t0 = prompt_check(txt, ['%Y%m%d','%Y%m%d%H%M'])
                            
                    txt = 'Indicate end time of the data you want to add (format YYYYMMDDHHMM, HHMM is optional) '
                    t1 = prompt_check(txt, ['%Y%m%d','%Y%m%d%H%M'])
                            
                    script_path = os.path.dirname(os.path.realpath(__file__)) 
                    default_config_path = Path(script_path, 'qpe', 
                                               'default_config.yml')
                                    
                    c = prompt_check('Enter location of the configuration file (in yml format): ',
                            check = 'yaml',
                            default = str(default_config_path))
                        
                    
                    success = False
                    while not success:
                        o = prompt('Enter the location where the generated files will be stored: ')
                        try:
                            if not os.path.exists(o):
                                os.makedirs(o)
                            assert(os.path.exists(o))
                            success = True
                        except:
                            pass
        
                    folder_models = Path(script_path, 'qpe', 
                                               'rf_models')
                    txt = 'Enter the name of the RF models to use in the form of a json line of the following format '+ \
                                   '{"model1_name":"model1_filename",model2_name":"model2_filename,...,modeln_name":"modeln_filename}' + \
                                   ', all model filenames must be stored in the folder ' + str(folder_models) + ' : '
                    m = prompt_check(txt, 'json')
                        
                    print('You want to compute a QPE with the following parameters...')
                    print('Starting time: ' + str(t0))
                    print('End time: ' + str(t1))
                    print('Output folder: ' + o)
                    print('Config file: ' + c)
                    print('Model(s): ' + str(m))
                    
                    ok = prompt('Do you want to start y/n: ')
                    if ok == 'y':
                        print('Creating slurm job')
                        fname = 'qpe.job'
                        file = open(fname,'w')
                        file.write(constants.SLURM_HEADER_PY)
                        file.write("qpe_compute -s {:s} -e {:s} -o {:s} -c {:s} -m '{:s}'".format(
                                   t0, t1, o, c, m))
                        file.close()
                        print('Submitting job')
                        subprocess.call('sbatch {:s}'.format(fname), shell = True)
                
        except KeyboardInterrupt:
            current_menu = 'main'
            print(title1, style=style)
            pass
            
