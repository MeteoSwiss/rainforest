#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command line interface to the database

Daniel Wolfensberger
MeteoSwiss/EPFL
daniel.wolfensberger@epfl.ch
December 2019
"""


# Global imports
import datetime
import os
from textwrap import dedent
from prompt_toolkit.shortcuts import prompt
from prompt_toolkit.formatted_text import HTML, ANSI, FormattedText
from prompt_toolkit.styles import Style
from prompt_toolkit import print_formatted_text

# Local imports
from rainforest.database.database import Database, DataFrameWithInfo

print = print_formatted_text

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


title = """-----------------------------------------
Radar-gauge database interface
Daniel Wolfensberger, LTE-MeteoSwiss, 2019
------------------------------------------"""

dbase = Database()

info = FormattedText([
        ('class:command','l'),
        ('',': load one or several new tables from files \n'),
        ('class:command','q'),
        ('',': run a new query \n'),
        ('class:command','a'),
        ('',': add results of last query to database \n'),
        ('class:command','p'),
        ('',': populate the database with new data \n'),
        ('class:command','s'),
        ('',': save results of last query to file \n'),        
        ('class:command','<table_name>'),
        ('',': displays info on a loaded table \n'),     
        ('class:command','e'),
        ('',': exit program \n')])

print(title, style=style)

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
            print(info,style=style_info)
        
        ########
        # Populate
        ########
        if code == 'p':
            n = None
            while n not in ['radar','reference','gauge']:
                n = prompt('With which type of data would you like to populate the database: "gauge", "radar" or "reference"? ')
            success = False
            while not success:
                if n == 'gauge':
                    txt = 'Indicate start time of the data you want to add (format YYYYMMDDHHMM) '
                else:
                    txt = 'Indicate start time of the data you want to add (format YYYYmmddHMM), leave empty to select automatically from gauge data: '
                
                t0 = prompt(txt)
                
                if n != 'gauge' and t0 == '':
                    success = True
                    t0 = None
                else:
                    try:
                        t0 = datetime.datetime.strptime(t0, '%Y%m%d%H%M')
                        success = True
                    except:
                        success = False
                        
            success = False
            while not success:
                if n == 'gauge':
                    txt = 'Indicate end time of the data you want to add (format YYYYMMDDHHMM) '
                else:
                    txt = 'Indicate end time of the data you want to add (format YYYYmmddHMM), leave empty to select automatically from gauge data: '
                
                t1 = prompt(txt)
                
                if n != 'gauge' and t1 == '':
                    success = True
                    t1 = None
                else:
                    try:
                        t1 = datetime.datetime.strptime(t1, '%Y%m%d%H%M')
                        success = True
                    except:
                        pass
            
            success = False
            if n != 'gauge':   
                while not success:
                    txt = dedent("""Select the gauge tables that will be used as a 
                    reference to select timesteps, indicate either the filepaths or 
                    the name of a table that has previously been added with the load
                    tables (l) instruction: 
                    """)
                    g = prompt(txt)
                    if g in dbase.tables.keys():
                        success = True
                    else:
                        try:
                            dbase.load_tables({'gauge': g}, False)
                            success = True
                        except:
                            pass
        
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
            
            success = False
            while not success:
                c = prompt('Enter location of the configuration file (in yml format): ')
                try:
                    dbase.config_file = c
                    success = True
                except:
                    pass
            
            print('You want to update the database with the following parameters...')
            print('Data type: '  + n)
            print('Starting time: ' + str(t0))
            print('End time: ' + str(t1))
            if n != 'gauge':
                print('Gauge reference: ' + g )
            print('Output folder: ' + o)
            print('Config file: '+c)
            
            ok = prompt('Enter to accept and start')
            
            if n == 'gauge':
                dbase.update_station_data(t0, t1, o)
            elif n == 'reference':
                dbase.update_reference_data('gauge', o, t0, t1)
            elif n == 'radar':
                dbase.update_radar_data('gauge',o, t0, t1)
                
        ########
        # Load
        ########
        if code == 'l':
            n = prompt('Enter name of table(s) (you choose), use comma to separate multiple entries: ', default = 'radar')
            d = prompt('Enter filepaths (ex. /mydir/*.csv) where the table(s) are stored, use comma to separate multiple entries: ', 
                       default = '/scratch/wolfensb/dbase_tests/radar/*.parquet')
    
            try:
                dic = {}
                for nn,dd in zip(n.split(','), d.split(',')):
                    dic[nn] = dd
                dbase.load_tables(dic)
                print('The table was successfully added', style = style_ok)
            except:
                print('Could not add table', style = style_warning)
        
        if code in dbase.tables.keys():
            print(dbase.tables[code].info)
            
        if code == 'a':
            if current_query == None:
                print('No query in memory!', style = style_warning)
            else:
                t =  prompt('Enter a table name for the query: ')
                dbase.tables[t] = DataFrameWithInfo(t, current_query)
        
        if code == 's':
            s =  prompt('Enter filename where to save query (.csv or .parquet): ')
            if current_query !=  None:
                df = current_query.toPandas()
                if '.csv' in s:
                    df.to_csv(s,index=False)
                elif '.parquet' in s:
                    df.to_parquet(s,index=False)
                else:
                    print('Invalid file type')
        if code == 'q':
            q =  prompt('Enter your SQL query: ')
            try:
                current_query = dbase.query(q)
                current_query.show(10)
            except Exception as e:
                print(e, style = style_warning)
    except KeyboardInterrupt:
        pass
        
