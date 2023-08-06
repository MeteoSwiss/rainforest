#!/usr/bin/env python3
"""
Jretrieve client using the REST interface.

You can set variables 'jretrieve_url' as the default URL and 'auth_header' for
the HTTP Authorization header in a file '~/.jretrievedwh-conf.py', for example
when using an offline token from https://servicedevt.meteoswiss.ch/api-token/:

```
import urllib.request
import urllib.parse
import json

jretrieve_url = 'https://servicedevt.meteoswiss.ch/jretrieve/api/v1'
with urllib.request.urlopen(urllib.request.Request(
        method='POST',
        url='https://authdevt.meteoswiss.ch/auth/realms/meteoswiss.ch/protocol/openid-connect/token',
        data=urllib.parse.urlencode((('grant_type', 'refresh_token'), ('client_id', 'mch-api'),
                                     ('refresh_token', 'PUT_YOUR_OFFLINE_TOKEN_HERE')
                                     )).encode())) as f:
    auth_header = 'Bearer ' + json.loads(f.read().decode())['access_token']
```

When using client credentials, you could use

```
import urllib.request
import urllib.parse
import base64
import json

jretrieve_url = 'https://servicedevt.meteoswiss.ch/jretrieve/api/v1'
with urllib.request.urlopen(urllib.request.Request(
        method='POST',
        url='https://servicedevt.meteoswiss.ch/auth/realms/meteoswiss.ch/protocol/openid-connect/token',
        data=b'grant_type=client_credentials',
        headers={b'Authorization': b'Basic ' + base64.b64encode(b'%s:%s' % (b'YOUR_CLIENT_ID', b'YOUR_CLIENT_SECRET'))}
)) as f:
    auth_header = 'Bearer ' + json.loads(f.read().decode())['access_token']
```
"""

import argparse
import numpy as np
import http.client
import logging
import os
import sys
import pandas as pd
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Dict, Union

# Set environment variable 'LOGLEVEL=DEBUG' to see the URLs
LOGLEVEL = os.environ.get('LOGLEVEL', 'CRITICAL').upper()
logging.basicConfig(level=LOGLEVEL)
logger = logging.getLogger('jretrieve')

NA_VALUE = 10000000.00 # value used by jretrieve for NA (missing value)

from rainforest.common.object_storage import ObjectStorage
ObjStorage = ObjectStorage()

def parse_arguments(argv: List[str]) -> Dict[str, str]:
    def fix_negative_coordinates(argv: List[str]) -> None:
        """
        Fix case of coordinates starting with a negative value, e.g. '-l "-90,90,-180,180"',
        which will raise 'expected one argument'. It will be changed to
        '-l=-90,90,-180,180', which is accepted.
        :param argv: argument list, will be modified
        :return:
        """
        try:
            coord_idx = argv.index('-l')
            coord_value = argv[coord_idx + 1]
            if coord_value.startswith('-'):
                argv[coord_idx] = '-l=' + coord_value
                del argv[coord_idx + 1]
        except (ValueError, IndexError):
            return

    # def type_string_list(arg: str) -> List[str]:
    #     return arg.split(',')
    #
    # def type_int_list(arg: str) -> List[int]:
    #     try:
    #         return [int(s) for s in arg.split(',')]
    #     except ValueError:
    #         raise argparse.ArgumentTypeError('%r is not list of integers' % arg)

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', help='print client version', action='version', version='2.0.0')
    parser.add_argument('-s', '--seq-type',
                        choices=['surface', 'surface_bufr', 'profile', 'profile_integral',
                                 'profile_angle_freq', 'profile_mobile', 'amdar',
                                 'profile_model', 'profile_model_integral', 'synop',
                                 'pointforecast', 'pointforecast_hist', 'pointforecast_poi',
                                 'pointforecast_poi_postal_code', 'pointforecast_poi_station',
                                 'extreme_value', 'norm', 'factor', 'aviation', 'aviation_deco',
                                 'mobile', 'mobile_buoy', 'mosworld'])  # default='surface'
    parser.add_argument('-v', '--verbose',
                        help='{integral{,parameter_id}{,parameter_id}|position}')  # type=type_string_list
    parser.add_argument('-t', '--time', help='time|mintime,maxtime{,incrementInMinutes}|MM[dd]')
    parser.add_argument('-i', '--locations',
                        help='{meas_site_id|net_nr|nat_ind|wmo_ind|int_ind|nat_abbr|'
                             'station_id|icao|wigos_id},loc_id{,loc_id}')
    parser.add_argument('-a', '--groups',
                        help='type as the first list element {meas_group|meas_group_id|'
                             'stn_group|stn_group_id},group{,group} or only list of '
                             'meas_group-names {group_name}{,group_name}')  # type=type_string_list
    parser.add_argument('-e', '--elevation', help='minelev,maxelev')
    parser.add_argument('-l', '--coordinates',
                        help='minlat,maxlat,minlong,maxlong. Use double-quote for negative '
                             'coordinates e.g. -l "-20,20,-30,30"')
    parser.add_argument('-j', '--info',
                        help='{lat,lon,elev,baro,name,net_nr,nat_ind,wmo_ind,int_ind,nat_abbr,'
                             'wigos_id,launch_ts (only Profile)}')  # type=type_string_list
    parser.add_argument('-c', '--meas-cat-nr', '--meas_cat_nr',
                        help='meas_cat_nr (>0)')  # type=int, default=1
    parser.add_argument('-C', '--data-source-id', '--data_source_id',
                        help='data_source_id (>0) (Profile)')  # type=int, default=1
    parser.add_argument('-w', '--obs-type-ids', '--obs_type_ids',
                        help='obs_type_id{,obs_type_id} (>0) (Profile)')  # type=type_int_list
    parser.add_argument('-W', '--prof-type-ids', '--prof_type_ids',
                        help='prof_type_id{,prof_type_id} (>0) (Profile)')  # type=type_int_list
    parser.add_argument('-p', '--parameter-ids', '--parameter_ids',
                        help='parameter_id{,parameter_id}')  # type=type_int_list
    parser.add_argument('-n', '--parameter-short-name', '--parameter_short_name',
                        help='short_name{,short_name}')  # type=type_string_list
    parser.add_argument('-r', '--quality-info', '--quality_info',
                        help='Show quality info. Bit array with PlausibilityNu-, '
                             'Mutation-Info-, DataQualityCatNr-flag; Range [0..15]; '
                             '0 (default): show no info, Bit 1: show DataQualityCatNr, '
                             'Bit 2: show Mutation-Info, Bit 3: show PlausibilityNu, '
                             'Bit 4: UncertaintyNu '
                             '(Profiles support only PlausibilityNu and UncertaintyNu')  # type=int, default=0
    parser.add_argument('--plausibility-limit', '--plausibility_limit',
                        help='Show only surface values with PlausibilityNu >= plausibilityLimit; '
                             'Range [0.0..1.0]')  # type=float
    parser.add_argument('--data-quality-cat-nr-limit', '--data_quality_cat_nr_limit',
                        help='Show only surface values with DataQualityCatNr >= '
                             'dataQualityCatNrLimit; Range [0: undefined | 1: impossible | '
                             '2: implausible | 3: suspect | 4: plausible]')  # type=int
    parser.add_argument('-m', '--model-run-time', '--model_run_time', help='Model run time interval')
    parser.add_argument('-u', '--url', help='Server URL')
    parser.add_argument('-debug', action='store_const', const='true', help=argparse.SUPPRESS)  # action='store_true'
    parser.add_argument('--encoding',
                        help='encoding used on client and server')  # default='UTF-8'
    parser.add_argument('--format', choices=['text', 'json', 'json-GeoJSON', 'csv', 'bufr'],
                        help='output format: text (default), json, json-GeoJSON, csv or bufr')  # default='text'
    parser.add_argument('--read-timeout', '--read_timeout',
                        help='read timeout in milliseconds')  # type=int
    parser.add_argument('--connect-timeout', '--connect_timeout',
                        help='connect timeout in milliseconds')  # type=int
    parser.add_argument('--profile-angle-freq-band', '--profile_angle_freq_band',
                        help='Profile-Angle-Freq parameter frequency '
                             '{freq_value|freq_min,freq_max}')  # type=type_string_list
    parser.add_argument('--profile-angle-freq-elevation', '--profile_angle_freq_elevation',
                        help='Profile-Angle-Freq parameter elevation '
                             '{elevation_value|elevation_min,elevation_max}')  # type=type_string_list
    parser.add_argument('--profile-angle-freq-azimut', '--profile_angle_freq_azimut',
                        help='Profile-Angle-Freq parameter azimut '
                             '{azimut_value|azimut_min,azimut_max}')  # type=type_string_list
    parser.add_argument('--level-nu', '--level_nu',
                        help='Profile: Optional LevelNu selection for profile request. '
                             '{level_nu|min_level_nu,max_level_nu}')
    parser.add_argument('--level-parameter-id', '--level_parameter_id',
                        help='Profile: Optional LevelParameterId option used together with '
                             '"--level_nu". {level_parameter_id}, default: 744')
    parser.add_argument('--mutation-info-nr', '--mutation_info_nr',
                        help='Mutation info number')  # type=int
    parser.add_argument('--aviation-info-options', '--aviation_info_options',
                        help='Aviation-Deco parameter aviation info options '
                             '{type|cor|valid_since|valid_till|ident}')  # type=type_string_list
    parser.add_argument('--aviation-bulletin-types', '--aviation_bulletin_types',
                        help='Aviation-Deco parameter bulletin types {type}{,type}, '
                             'spaces must be replaced with \'_\' e.g. METAR,AIRMET,TAF_LONG')  # type=type_string_list
    parser.add_argument('--meta-info', '--meta_info',
                        help='Station or measuring field infos {data_owner|use_limitation|lat|'
                             'lon|elev|baro|name|nat_abbr|nat_ind|net_nr|wmo_ind|int_ind}')  # type=type_string_list
    parser.add_argument('--use-limitation', '--use_limitation',
                        choices=['10', '20', '30', '40', '50'],
                        help='Limitation value, only surface data with '
                             'use_limitation <= limit will be returned.'
                             '  10=Use without any limitation'
                             ', 20=Use without any limitation, reference always required'
                             ', 30=Limited use, transfer to Swiss task forces and research '
                             'allowed, reference always required'
                             ', 40=Only internal use, transfer to Swiss task forces allowed, '
                             'reference always required'
                             ', 50=Only internal use, no transfer to anybody')  # type=int, default = 20
    parser.add_argument('-d', '--delimiter',
                        help='Delimiter in the output data {CSV|PIPE|COMMA|TAB|text between ".."}')
    parser.add_argument('-x', '--placeholder',
                        help='Placeholder for nan-values any text e.g. 10000000,"-","NA"')
    parser.add_argument('--header-disabled', '--header_disabled', action='store_const', const='true',
                        help='Disable header line, return only data')  # action='store_true'
    parser.add_argument('-b', '--reference-period', '--reference_period',
                        choices=['1', '2', '3', '4'],
                        help='Reference period of extreme_value. '
                             '1 (default): before 1959, 2: since 1959, 3: since 1978')  # type=int, default=1
    parser.add_argument('-B', '--effective-reference-periods', '--effective_reference_periods',
                        choices=['y', 'n', 'true', 'false'],
                        help='Show information about effective reference periods for extreme_value.')  # default='n'
    parser.add_argument('-R', '--number-of-rankings', '--number_of_rankings',
                        help='{y|true|n|false|1..10} '
                             'Number of ranks which will display at extreme_value, '
                             'y|true: show all 10 ranks, n|false (default): show only top rank, '
                             '1..10: show this number of ranks')  # default='n'
    parser.add_argument('-g', '--granularity',
                        choices=['Y', 'M', 'D'],
                        help='Granularity option for norm and factor; Y=Year, M=Month, '
                             'D=Daily (default)')  # default='D'
    parser.add_argument('--show-records', '--show_records', action='store_const', const='true',
                        help='Show number of records at end of output data')  # action='store_true'
    parser.add_argument('--profile-time-offset', '--profile_time_offset',
                        help='Profile-Mobile, Amdar: Time offset in minutes for profile-lookup (>=0),'
                             ' default=30')  # type=int, default=30
    parser.add_argument('-o', '--output',
                        help='Write output to file.')

    fix_negative_coordinates(argv)
    if len(argv) == 0:
        parser.print_help(sys.stderr)
        sys.exit(1)
    return vars(parser.parse_args(argv))


def exec_conf_file() -> Dict[str, str]:
    data_folder = os.environ['RAINFOREST_DATAPATH']
    conf_file_path = str(Path(data_folder, 'credentials', '.jretrievedwh-conf.py'))
    ObjStorage.check_file(conf_file_path)
    cfg = {}
    if os.path.exists(conf_file_path):
        with open(conf_file_path) as fh:
            script = fh.read()
            exec(script, {}, cfg)
    return cfg


def to_camel_case(snake_str: str) -> str:
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


def retrieve(args: Dict[str, str], cfg: Dict[str, str], return_body: bool = False) -> int:
    def extract_view_type(params: Dict[str, str], param_name: str,
                          view_types: Union[List[str], Dict[str, str]],
                          modify_params = True):
        param_values = params.get(param_name)
        if param_values:
            comma_pos = param_values.find(',')
            if comma_pos != -1:
                try:
                    # noinspection PyTypeChecker
                    view_type = view_types.get(param_values[:comma_pos], None)
                except AttributeError:
                    view_type = param_values[:comma_pos]
                    if view_type not in view_types:
                        view_type = None
                if view_type is not None:
                    # The first element is the view type
                    path = params.get('path', '')
                    if path == 'mosworld':
                        return
                    elif path == 'surface/bufr':
                        path = 'surface/' + view_type + '/bufr'
                        params['path'] = path
                    elif path.startswith(('profile/', 'profile_model/')) or path in ('surface', 'synop'):
                        path += '/' + view_type
                        params['path'] = path
                    else:
                        params['viewType'] = view_type
                    if modify_params:
                        param_values = param_values[comma_pos + 1:]
                        params[param_name] = param_values

    def fix_parameter_values(params: Dict[str, str]):
        aviation_bulletin_types = params.get('aviationBulletinTypes')
        if aviation_bulletin_types is not None:
            # 'TAF_LONG' must be 'TAF LONG'
            params['aviationBulletinTypes'] = aviation_bulletin_types.replace('_', ' ')
        extract_view_type(params, 'locationIds', [
            'installation', 'station_id', 'nat_abbr', 'nat_ind', 'wmo_ind', 'int_ind',
            'meas_site_id', 'net_nr', 'icao', 'postal_code_nr', 'postal_code_id', 'wigos_id'])
        extract_view_type(params, 'locationGroup', {
            'meas_group': 'meas_site_id', 'meas_group_id': 'meas_site_id',
            'stn_group': 'station_id', 'stn_group_id': 'station_id'}, modify_params=False)
        if params.get('path') in ('profile/integral', 'profile/data', 'profile/angle_freq'):
            # To support queries with coordinates instead of location IDs
            params['path'] += '/wmo_ind'
        yn_value = params.get('effectiveRefPeriod')
        if yn_value in ('y', 'n'):
            params['effectiveRefPeriod'] = 'true' if yn_value == 'y' else 'false'
        yn_value = params.get('rankingNr')
        if yn_value in ('y', 'n'):
            params['rankingNr'] = '10' if yn_value == 'y' else '1'

    show_records = (args.pop('show_records', '') or '').lower() == 'true'  # type: bool
    host = args.pop('url')
    if not host:
        host = cfg['jretrieve_url'] if 'jretrieve_url' in cfg \
            else 'https://jretrieve-prod.apps.cp.meteoswiss.ch/jretrieve/api/v1'
    host = host.rstrip('/')

    connect_timeout_ms = 0
    connect_timeout_str = args.pop('connect_timeout', '')
    if connect_timeout_str:
        try:
            connect_timeout_ms = int(connect_timeout_str)
        except ValueError:
            raise ValueError('Invalid parameter connect_timeout: ' + connect_timeout_str)

    read_timeout_ms = 0
    read_timeout_str = args.pop('read_timeout', '')
    if read_timeout_str:
        try:
            read_timeout_ms = int(read_timeout_str)
        except ValueError:
            raise ValueError('Invalid parameter read_timeout: ' + read_timeout_str)

    output_file_name = args.pop('output')

    params = {}  # type: Dict[str, str]
    seq_type = args.pop('seq_type', None)
    if seq_type:
        params['path'] = {
            'profile_integral': 'profile/integral',
            'profile': 'profile/data',
            'profile_angle_freq': 'profile/angle_freq',
            'profile_model': 'profile_model/data',
            'profile_model_integral': 'profile_model/integral',
            'mobile': 'synop_mobile',
            'mobile_buoy': 'synop_mobile_buoy',
            'surface_bufr': 'surface/bufr',
            'pointforecast_poi_station': 'pointforecast_poi/station',
            'pointforecast_poi_postal_code': 'pointforecast_poi/postal_code'
        }.get(seq_type, seq_type)
    if args.get('meta_info'):
        params['path'] = 'meta_info'

    name_map = {
        'coordinates': 'coord',
        'effective_reference_periods': 'effectiveRefPeriod',
        'groups': 'locationGroup',
        'info': 'infoOptions',
        'meta_info': 'infoOptions',
        'locations': 'locationIds',
        'model_run_time': 'modelrunTsInterval',
        'number_of_rankings': 'rankingNr',
        'parameter_short_name': 'parameterShortNames',
        'profile_angle_freq_azimut': 'azimut',
        'profile_angle_freq_band': 'frequency',
        'profile_angle_freq_elevation': 'elevation',
        'quality_info': 'quality',
        'reference_period': 'refPeriodId',
        'time': 'date'
    }
    for key, value in args.items():
        if value is not None:
            name = name_map.get(key, to_camel_case(key))
            params[name] = value
    fix_parameter_values(params)

    path = params.pop('path', '')
    url = host + '/' + path + '?' + urllib.parse.urlencode(params)
    logger.debug('GET ' + url)
    timeout_ms = connect_timeout_ms + read_timeout_ms
    timeout = timeout_ms / 1000 if timeout_ms > 0 else None
    req = urllib.request.Request(url=url)
    if 'auth_header' in cfg:
        req.add_header('Authorization', cfg['auth_header'])
    try:
        with urllib.request.urlopen(req, timeout=timeout) as f:  # type: http.client.HTTPResponse
            body = f.read()
            if body.startswith(b'<!DOCTYPE html') and b'<title>Login' in body:
                # Probably an OIDC redirect from Keycloak
                raise urllib.error.URLError('Login required')
            if output_file_name:
                with open(output_file_name, 'wb') as output_fh:
                    output_fh.write(body)
            else:
                sys.stdout.buffer.write(body)
            if show_records:
                try:
                    num_records = int(f.headers.get('RECORDS'))
                except TypeError:
                    stripped_body = body.strip()
                    num_records = len(stripped_body.split(b'\n')) if stripped_body else 0
                    if not params.get('headerDisabled'):
                        num_records -= 2
                        if num_records < 0:
                            num_records = 0
                sys.stdout.buffer.write(b'records read: %d%s' % (num_records, os.linesep.encode()))
    except urllib.error.HTTPError as e:
        body = e.read()
        if body:
            sys.stderr.buffer.write(b'ERROR %d: %s%s' % (e.code, body, os.linesep.encode()))
        logger.exception(body.decode())
        return 1
    if return_body:
        return body
    else:
        return 0


def main(argv: List[str] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    logger.debug('ARG ' + ' '.join(argv))
    args = parse_arguments(argv)
    cfg = exec_conf_file()
    return retrieve(args, cfg)

def jretrievedwh_to_pandas(argv: List[str] = None) -> pd.DataFrame:
    if argv is None:
        argv = sys.argv[1:]
    logger.debug('ARG ' + ' '.join(argv))
    args = parse_arguments(argv)

    cfg = exec_conf_file()
    out = retrieve(args, cfg, return_body=True)
    colnames = out.split(b'\n')[1].decode('utf-8').split()
    lines = out.split(b'\n')[3:-1]
    split_lines = [l.decode('utf-8').replace(' ','').split('|') for l in lines]
    
    df = pd.DataFrame(split_lines, columns = colnames)
    df = df.rename(columns={"termin": "datetime"})
    df['datetime'] =  pd.to_datetime(df['datetime'])
    for col in df.columns:
        if col == 'datetime':
            continue
        df[col] = pd.to_numeric(df[col])
    df = df.replace(NA_VALUE, np.nan)
    return df

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

