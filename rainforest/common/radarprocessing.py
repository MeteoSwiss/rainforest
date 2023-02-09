#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 17:38:52 2019
@author: wolfensb, gugerlir
"""


import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from textwrap import  dedent
import datetime

from pyart.retrieve import kdp_leastsquare_single_window
from pyart.retrieve import hydroclass_semisupervised
from pyart.retrieve import compute_noisedBZ
from pyart.correct import smooth_phidp_single_window
from pyart.correct import calculate_attenuation_zphi
from pyart.aux_io import read_metranet, read_cartesian_metranet
from pyart.testing import make_empty_ppi_radar
from pyart.core.transforms import antenna_vectors_to_cartesian

from .logger import logger
from .utils import sweepnumber_fromfile, rename_fields
from .io_data import read_status, read_vpr
from . import constants
from . import wgs84_ch1903
from .lookup import get_lookup

class Radar(object):
    '''
    A class that contains polar radar data and performs some pre-processing
    before adding that data to the database or computing the QPE product
    The different elevations are stored in a dictionary rather as in a
    single pyart radar instance as this was found to be faster in practice
    '''
    def __init__(self, radname, polfiles, statusfile=None, vprfile=None,
                temp_ref='TAIR', metranet_reader = 'python'):
        """
        Creates an Radar class instance
        
        Parameters
        ----------
        radname : char
            Name of the radar, either 'A','D','L','P' or 'W'
        polfiles : list of str
            List of full filepaths of the radar files for a given timestemp,
            one file for every elevation, typically obtained with
            the retrieve_prod function in the common submodule
        statusfile: str (optional)
            Full path of the status file that corresponds to this particular
            radar and timestep, used to compute noise estimates
        vprfile : str(optional)
             Full path of the vpr xml file that corresponds to this particular
            radar and timestep, used to compute VPR correction
        metranet_reader : str(optional)
             Which reader to use to read the polar radar data, can be either 'C' or 'python'
        """
        
        self.sweeps = []
        self.radsweeps = {}
        
        visib = get_lookup('visibility_rad', radname)
    
        for f in polfiles:
            try:
                sweep = sweepnumber_fromfile(f)
                radinstance = read_metranet(f, reader = metranet_reader)
                rename_fields(radinstance)
                visib_sweep = np.ma.array(visib[sweep].astype(np.float32), 
                                          mask = np.isnan(visib[sweep]))

                zh = radinstance.get_field(0,'ZH')
                visib_sweep = visib_sweep[0:len(zh),:]
                radinstance.add_field('VISIB',{'data': visib_sweep})
                                
                self.radsweeps[sweep] = radinstance
                self.sweeps.append(sweep)
            except:
                logger.error('Could not read file {:s}'.format(f))
                pass

        self.radname = radname
        
        if statusfile != None:
            try:
                # For the rare occasion that more than one statusfile for this radar and time exists
                if type(statusfile) == list:
                    statusfile = statusfile[0]   
                self.status =  read_status(statusfile)
                self.compute_noise()
            except:
                logger.error('Could not compute noise from status file!')
                pass
            
        if vprfile != None:
            try:
                self.vpr = read_vpr(vprfile, self.radname)
            except:
                logger.error('Could not add vpr file!')
                pass

        # To get a variable to define the temperature reference
        self.temp_ref = temp_ref

        # To keep track of the nature of data fields
        try:
            self.radarfields = list(self.radsweeps[self.sweeps[0]].fields.keys())
            self.cosmofields = [] # updated later
            self.precipfield  = [] # updated later
        except:
            self.radarfields = []
              
    def snr_mask(self, snr_threshold):
        """
        Masks the radar data at low SNR
        
        Parameters
        ----------
        snr_threshold : float
            Minimal SNR to consider in dB
        """
        
        for s in self.sweeps:
            radsweep = self.radsweeps[s]
                            
            if 'NH' not in radsweep.fields:
                msg = '''Could not find NH (noise) field in radar instance, 
                         please run first compute_noise()'''
                         
                raise ValueError(dedent(msg))
                
            snr = (radsweep.fields['ZH']['data'] - 
                   radsweep.fields['NH']['data'])

            # Mask data below SNR and with visib < threshold
            masked = snr < snr_threshold
            
            for k in self.radarfields: # Apply only to radar data, COSMO not affected
                radsweep.fields[k]['data'].mask[masked] = True
        
    def visib_mask(self, min_visib, max_visib_corr):
        """
        Masks the radar data at low visibility and corrects the reflectivity
        for visibility
        
        Parameters
        ----------
        min_visib : int
            Minimal visibility below which the data is masked
        max_visib_corr : float
            Maximum visibility correction factor, the visibility correction 
            is 100/VISIB (with VISIB in %) and can be thresholded with this 
            parameter. This is usually set to 2 at MeteoSwiss
        """
        
        for s in self.sweeps:
            radsweep = self.radsweeps[s]
            visib = radsweep.fields['VISIB']['data']

            z = radsweep.fields['ZH']['data']
            zlin = 10 ** (0.1 * z)
            corr = 1./(visib/100.)
            corr[corr >= max_visib_corr] = max_visib_corr
           
            zlin_corr = zlin * corr
            # mask
            zlin_corr.mask[visib < min_visib ] = True
            radsweep.add_field('ZH_VISIB',{'data': 10 * np.log10(zlin_corr)})
            

            z = radsweep.fields['ZV']['data']
            zlin = 10 ** (0.1 * z)
            corr = 1. / (visib / 100.)
            corr[corr >= max_visib_corr] = max_visib_corr

            zlin_corr = zlin * corr
            # mask
            zlin_corr.mask[visib < min_visib ] = True
            radsweep.add_field('ZV_VISIB',{'data':  10 * np.log10(zlin_corr)})

    def compute_noise(self):
        """
        Computes a noise estimate from a status file
        """
        
        for i,s in enumerate(self.sweeps):
            radsweep = self.radsweeps[s]
            try:
                noise_h = float(self.status['status']['sweep'][i]['RADAR']['STAT']
                            ['CALIB']['noisepower_frontend_h_inuse']['@value'])
                rconst_h = float(self.status['status']['sweep'][i]['RADAR']['STAT']
                                            ['CALIB']['rconst_h']['@value'])
                noisedBADU_h = 10.*np.log10(noise_h) + rconst_h
                
                noise_v = float(self.status['status']['sweep'][i]['RADAR']['STAT']
                                ['CALIB']['noisepower_frontend_v_inuse']['@value'])
                rconst_v = float(self.status['status']['sweep'][i]['RADAR']['STAT']
                                        ['CALIB']['rconst_v']['@value'])
                noisedBADU_v = 10.*np.log10(noise_v) + rconst_v
                
            except:
                # default noise
                noisedBADU_h = constants.NOISE_100
                noisedBADU_v = constants.NOISE_100
                pass
            
            noisedBZ_h = compute_noisedBZ(radsweep.nrays, noisedBADU_h,
                    radsweep.range['data'], 100.,
                    noise_field='noisedBZ_hh')
                
            noisedBZ_v = compute_noisedBZ(radsweep.nrays, noisedBADU_v,
                    radsweep.range['data'], 100.,
                    noise_field='noisedBZ_vv')
            
            # Convert to masked array for consistency
            noisedBZ_h['data'] = np.ma.array(noisedBZ_h['data'], 
                                        mask = np.isnan(noisedBZ_h['data'])) 
            noisedBZ_v['data'] = np.ma.array(noisedBZ_v['data'], 
                                        mask = np.isnan(noisedBZ_v['data']))    
            
            radsweep.add_field('NH', noisedBZ_h)
            radsweep.add_field('NV', noisedBZ_v)

    def add_cosmo_data(self, cosmo_data):
        """
        Adds COSMO data to the radar instance
        
        Parameters
        ----------
        cosmo_data : dict
            dict of COSMO data at polar coordinates obtained from the 
            get_COSMO_variables function in the common submodule
            this dict must have the following format
                dic[variable][sweep]
        """
        
        all_vars = cosmo_data.keys()
        
        for v in all_vars:
            # Take only cosmo data for the sweeps we have
            for s in self.sweeps:
                cdata = cosmo_data[v][s].copy() # deepcopy, important
                if v == 'T':
                    cdata -= 273.15 # COnvert to celcius

                self.radsweeps[s].add_field(v, {'data': cdata})
                self.cosmofields.append(v)
            
    def compute_hydro(self):
        """
        Computes the hydrometeor classification using Nikola Besic' 
        algorithm, all necessary fields 
        ZH, ZDR, RHOHV, KDP, T (COSMO) must be available
        """
        
        for s in self.sweeps:
            radsweep = self.radsweeps[s]
            out = hydroclass_semisupervised(radsweep, refl_field = 'ZH',
                                            zdr_field = 'ZDR',
                                            rhv_field = 'RHOHV',
                                            kdp_field = 'KDP',
                                            temp_ref = 'temperature',
                                            temp_field = 'T',
                                            vectorize = True)
            
            radsweep.add_field('HYDRO', out['hydro'])
        
    def correct_attenuation(self):
        """
        Corrects for attenuation using the ZPHI algorithm (Testud et al.)
        using the COSMO temperature to identify liquid precipitation
        OR using the 0° isothermal altitude
        """
        
        for s in self.sweeps:
            radsweep = self.radsweeps[s]

            if self.temp_ref == 'TAIR':
                ah, pia, cor_z, _, pida, cor_zdr = calculate_attenuation_zphi(
                                radsweep,
                                refl_field='ZH',
                                zdr_field = 'ZDR',
                                phidp_field = 'PHIDP',
                                temp_field = 'T',
                                temp_ref = 'temperature',
                                doc = 15)
            elif self.temp_ref == 'ISO0_HEIGHT':
                ah, pia, cor_z, _, pida, cor_zdr = calculate_attenuation_zphi(
                                radsweep,
                                refl_field='ZH',
                                zdr_field = 'ZDR',
                                phidp_field = 'PHIDP',
                                iso0_field = 'height_over_iso0',
                                temp_ref = 'height_over_iso0',
                                doc = 15)

            radsweep.add_field('AH', ah)
            radsweep.add_field('ZH_CORR', cor_z)
            radsweep.add_field('ZDR_CORR', cor_zdr)
            
            zv_corr = pia['data'] - pida['data'] + radsweep.get_field(0, 'ZV')
            radsweep.add_field('ZV_CORR', {'data': zv_corr})
            
    def compute_kdp(self, dscfg):
        """
        Computes KDP using the simple moving least-square algorithm
        
        Parameters
        ----------
        dscfg : dict
            dictionary containing the following fields
            RMIN: 
            RMAX: 
            RWIND: 
            ZMIN: 
            ZMAX: 
        """
        
        for s in self.sweeps:
            radsweep = self.radsweeps[s]
            ind_rmin = np.where(radsweep.range['data'] > dscfg['RMIN'])[0][0]
            ind_rmax = np.where(radsweep.range['data'] < dscfg['RMAX'])[0][-1]
            r_res = radsweep.range['data'][1]-radsweep.range['data'][0]
            min_rcons = int(dscfg['RCELL']/r_res)
            wind_len = int(dscfg['RWIND']/r_res)
            min_valid = int(wind_len/2+1)

            psidp_field = 'PSIDP'
            refl_field = 'ZH'
            phidp_field = 'PHIDP'
        
            phidp = smooth_phidp_single_window(
                radsweep, ind_rmin=ind_rmin, ind_rmax=ind_rmax, min_rcons=min_rcons,
                zmin=dscfg['ZMIN'], zmax=dscfg['ZMAX'], wind_len=wind_len,
                min_valid=min_valid, psidp_field=psidp_field, refl_field=refl_field,
                phidp_field=phidp_field)
            
            radsweep.add_field(phidp_field,phidp)
            
            r_res = radsweep.range['data'][1] - radsweep.range['data'][0]
            wind_len = int(dscfg['RWIND']/r_res)
            min_valid = int(wind_len/2+1)
            kdp_field = 'KDP'
            
            kdp = kdp_leastsquare_single_window(
                radsweep, wind_len=wind_len, min_valid=min_valid, 
                phidp_field=phidp_field, kdp_field=kdp_field, 
                vectorize = True)
  
         
            radsweep.add_field('KDP', kdp)
    
    def get_field(self, sweep, field_name):
        """
        Gets a radar variable at given elevation (sweep)
        
        Parameters
        ----------
        sweep : int
            Sweep number from 1 to 20
        field_name: str
            name of the variable, e.g. ZH, ZDR, RHOHV, SW, ...
        """
        
        # Check if all uppercase
        field_name_upper = field_name.upper()
        data = self.radsweeps[sweep].get_field(0, field_name_upper)
        # Convention is lowercase is radar variable in linear scale
        if field_name_upper != field_name:
            data = 10 ** (0.1 * data)
        return data

    def correct_gate_altitude(self):
        """
        Following the Swiss standard, we change the calculation of
        the gate_altitude that is automatically done in pyart.
        Instead of the constant radar scale factor of (ke) of 4/3, 
        we now use 1.25
        """

        for s in self.sweeps:
            radsweep = self.radsweeps[s]
            _,_,heights = antenna_vectors_to_cartesian(radsweep.range['data'],
                                radsweep.azimuth['data'], radsweep.elevation['data'], ke = 1.25)

            radsweep.gate_altitude['data'] = heights + radsweep.altitude['data']


    def add_hzt_data(self, hzt_cart):
        """
            Transform a Cartesian HZT object (height of freezing level) and
            adds the fields to this radar object.
            
            Parameters
            ----------
            hzt : Masked array as output from HZT_hourly_to_5min
                  from Grid object that contains the hzt data, as obtained with
                  pyart.aux_io.read_cartesian_metranet(filelist[tstamp_hzt0]).fields['iso0_height']['data'][0]

        """

        # Get lookup table for the radar
        lut = get_lookup('qpegrid_to_rad', self.radname)
    
        # Get Swiss coordinates
        CHX = constants.X_QPE
        CHY = constants.Y_QPE        
        
        for s in self.sweeps:
            radsweep = self.radsweeps[s]

            # lookup tables the sweeps are labelled 0-19
            # while in self.sweeps, they range from 1-20
            lut_sweep = lut[lut[:,0] == (s-1)]
            nrange = lut_sweep[:,2].max() + 1 
            naz = lut_sweep[:,1].max() + 1
        
            # Get Cartesian and polar indexes
            idxx = (lut_sweep[:,-1] - CHX[-1]).astype(int) - 1 
            idxy = (lut_sweep[:,-2] - CHY[0]).astype(int) - 1
        
            idxaz = lut_sweep[:,1]
            idxrange = lut_sweep[:,2]
        
            # Initialize polar arrays
            hzt_pol = np.zeros((naz, nrange))
            npts = np.zeros((naz, nrange))
        
            # Get part of Cart HZT that covers radar
            toadd = hzt_cart[idxx.ravel(), idxy.ravel()]
            
            # update grid
            hzt_pol[idxaz.ravel(), idxrange.ravel()] += toadd
            npts[idxaz.ravel(), idxrange.ravel()] += np.ones(toadd.shape)
            
            # To avoid a division trhough 0, which causes a python runtime warning:
            npts[npts == 0] = np.nan

            hzt_pol /= npts
        
            # Fill holes with nearest neighbour interpolation
            x,y=np.mgrid[0:naz, 0:nrange]
            
            xygood = np.array((x[~np.isnan(hzt_pol)],
                            y[~np.isnan(hzt_pol)])).T
            xybad = np.array((x[np.isnan(hzt_pol)],
                            y[np.isnan(hzt_pol)])).T
            
            hzt_pol[np.isnan(hzt_pol)] = hzt_pol[~np.isnan(hzt_pol)][
                KDTree(xygood).query(xybad)[1]]
        
            # Assure same sized fields
            hzt_pol_field = np.zeros((radsweep.nrays, radsweep.ngates)) + np.nan
            hzt_pol_field[:,0:hzt_pol.shape[1]] = hzt_pol

            hzt_dict = {'data':hzt_pol_field, 'units':'m', 
                    'long_name':'Height of freezing level',
                    'standard_name' :'HZT'}

            radsweep.add_field('ISO0_HEIGHT', hzt_dict)
        
        self.cosmofields.append('ISO0_HEIGHT')

    def add_height_over_iso0(self):
        """ Function to derive the relative height over the 0° isothermal altitude
            in meters by subtracting the altitude of the 0° isothermal altitude from 
            the altitude of the radar gate
        """

        # Correct the automatically derived gate altitude
        self.correct_gate_altitude()

        # Add the height over iso0 to the sweeps
        for s in self.sweeps:
            radsweep = self.radsweeps[s]
            height_over_iso0 = radsweep.gate_altitude['data']-radsweep.fields['ISO0_HEIGHT']['data']

            iso0_dict = {'data':height_over_iso0, 'units':'m', 
            'long_name':'height of freezing level with respect to radar gate altitude',
            'standard_name' :'height_over_iso0'}
            
            radsweep.add_field('height_over_iso0', iso0_dict)

        self.cosmofields.append('height_over_iso0')

def HZT_hourly_to_5min(time,filelist,tsteps_min=5):
    """ Function to interpolate the hourly isothermal fields to 5min resolution
        to make them consistant with the radar fields 

        Parameters
        ----------
        time : datetime object
            timestep to calculate
        filelist : dictionnary
            list with timesteps, path and filename of HZT files only
            typically derived in database.retrieve_radar_data.Updater.retrieve_radar_files()
        tsteps_min: int
            resolution of new fields in minutes

        Returns
        ----------
        dictionnary with datetime objects as keys and numpy.ma.core.MaskedArray (Cartesian coordinates)
    """
    if time.minute != 0:
        # Minutes are set to 0 below, here is only a notification
        logger.info('ISO0_HEIGHT: Temporal interpolation, timestamp {} set to HH:00'.format(time))

    tstamp_hzt0 = datetime.datetime(time.year, time.month, time.day, time.hour,0)
    tstamp_hzt1 = tstamp_hzt0+ datetime.timedelta(hours=1)

    hzt = {}
    hzt[tstamp_hzt0] = read_cartesian_metranet(filelist[tstamp_hzt0]).fields['iso0_height']['data'][0]
    hzt[tstamp_hzt1] = read_cartesian_metranet(filelist[tstamp_hzt1]).fields['iso0_height']['data'][0]

    # Giving info about which files are used
    logger.info('ISO0_HEIGHT: Temporal interpolation between {} and {}'.format(filelist[tstamp_hzt0], filelist[tstamp_hzt1]))

    # Get the incremental difference for e.g. 5min steps (divided by 12):
    dt = datetime.timedelta(minutes=tsteps_min)
    ndt = np.arange(1,int(60/tsteps_min))
    deltaHZT = (hzt[tstamp_hzt1]-hzt[tstamp_hzt0])/ (len(ndt)+1)

    # Loop through all min increments and add the calculated increment of deltaHZT
    for idx in ndt:
        if idx == ndt[0]:
            deltaHZT_temp = deltaHZT.copy()
        else:
            deltaHZT_temp += deltaHZT
        hzt[tstamp_hzt0+dt*idx] = hzt[tstamp_hzt0]+deltaHZT_temp

    return hzt

def HZT_cartesian_to_polar(hzt, radar, sweeps=range(0, 20)):
    """
        Transform a Cartesian HZT object (height of freezing level) to a polar
        Radar object with the freezing level at every gate on the Rad4Alp
        domain
        
        Parameters
        ----------
        hzt : pyart Grid object
            Grid object that contains the hzt data, as obtained with
            pyart.aux_io.read_cartesian_metranet
        radar: char
            name of the radar, 'A','D','L','P' or 'W'
        sweeps: list
            list of Rad4Alp sweeps, from 0 to 20 to include in the polar radar
            object
    """
        
        
    gps_converter = wgs84_ch1903.GPSConverter()

    hzt_cart = hzt.fields['iso0_height']['data'][0]

    lut = get_lookup('qpegrid_to_rad', radar)
    
    CHX = constants.X_QPE
    CHY = constants.Y_QPE
    
    hzt_pol_all_sweeps = []
    for s in sweeps:
        lut_sweep = lut[lut[:,0] == s]
        nrange = lut_sweep[:,2].max() + 1 
        naz = lut_sweep[:,1].max() + 1
        
        # Get Cartesian and polar indexes
        idxx = (lut_sweep[:,-1] - CHX[-1]).astype(int) - 1 
        idxy = (lut_sweep[:,-2] - CHY[0]).astype(int) - 1
        
        idxaz = lut_sweep[:,1]
        idxrange = lut_sweep[:,2]
        
        # Initialize polar arrays
        hzt_pol = np.zeros((naz, nrange))
        npts = np.zeros((naz, nrange))
        
        # Get part of Cart HZT that covers radar
        toadd = hzt_cart[idxx.ravel(), idxy.ravel()]
        
        # update grid
        hzt_pol[idxaz.ravel(), idxrange.ravel()] += toadd
        npts[idxaz.ravel(), idxrange.ravel()] += np.ones(toadd.shape)
        
        # To avoid a division trhough 0, which causes a python runtime warning:
        npts[npts == 0] = np.nan
    
        hzt_pol /= npts
        
        # Fill holes with nearest neighbour interpolation
        x,y=np.mgrid[0:naz, 0:nrange]
        
        xygood = np.array((x[~np.isnan(hzt_pol)],
                           y[~np.isnan(hzt_pol)])).T
        xybad = np.array((x[np.isnan(hzt_pol)],
                          y[np.isnan(hzt_pol)])).T
        
        hzt_pol[np.isnan(hzt_pol)] = hzt_pol[~np.isnan(hzt_pol)][
            KDTree(xygood).query(xybad)[1]]
        
        hzt_pol_all_sweeps.append(hzt_pol)
    
    # Make radar object
    nrays_per_sweep = hzt_pol_all_sweeps[0].shape[0]
    ngates = hzt_pol_all_sweeps[0].shape[1]
    hztradar = make_empty_ppi_radar(ngates, nrays_per_sweep, len(sweeps))
    
    # Create single array from all sweeps
    hzt_pol_field = np.zeros((nrays_per_sweep * len(sweeps), ngates)) + np.nan
    for i in range(len(hzt_pol_all_sweeps)):
        hzt_pol_field[i * nrays_per_sweep: (i+1) * nrays_per_sweep,
                      0:hzt_pol_all_sweeps[i].shape[1]] = hzt_pol_all_sweeps[i]
    hzt_dic = {'data':hzt_pol_field, 'units':'m', 
               'long_name':'Height above freezing level',
               'standard_name' :'HZT', 'coordinates':'elevation azimuth range'}
    hztradar.add_field('iso0_height', hzt_dic)
    hztradar.range['data'] = np.arange(len(hztradar.range['data'])) * 500 + 250

    # Add radar coordinates
    radar_pos = constants.RADARS[constants.RADARS.Abbrev == radar]
    radar_lat = gps_converter.CHtoWGSlat(radar_pos.Y, radar_pos.X)
    radar_lon = gps_converter.CHtoWGSlng(radar_pos.Y, radar_pos.X)
    hztradar.longitude['data'] = np.array([float(radar_lon)])
    hztradar.latitude['data'] = np.array([float(radar_lat)])
    hztradar.altitude['data'] = np.array([float(radar_pos.Z)])
    
    return hztradar
    

def hydroClass_single(radars, zh, zdr, kdp, rhohv, temp, 
                      weights = np.array([1., 1., 1., 0.75, 0.5])):
    """
    Computes the hydrometeor classes for columnar data, note that all input
    arrays except weights must have the same length
    
    Parameters
    ----------
    radars : ndarray of char
        Array of radar IDs, ('A','D','L','P','W')
    zh : ndarray
        Array of radar reflectivity in dBZ
    zdr: ndarray
        Array of diff. reflectivity in dB
    kdp: ndarray
        Array of specific phase shift on propagation in deg / km
    rhohv: ndarray
        Array of copolar correlation coefficient
    temp: ndarray
        Array of radar temperature in Celcius
    weights: ndarray (optional)
        The weight of every input feature, zh, zdr, kdp, rhohv, temp in the
        hydrometeor classification
           
    Returns
    -------
    The hydrometeor classes as ndarray with values from 0 to 8, corresponding to
    the classes
        0 : no data
        1 : aggregates (AG)
        2 : light rain (LR)
        3 : ice crystals (CR)
        4 : rimed particles (RP)
        5 : rain (RN)
        6 : vertically aligned ice (VI)
        7 : wet snow (WS)
        8 : melting hail (MH)
        9: dry hail / high density graupel (IH/HDG)
    """
    
    from pyart.retrieve.echo_class import _standardize
    unique_radars = np.unique(radars)
    hydro = np.zeros(len(radars)) + np.nan
    
    for r in unique_radars:
        
        idx = np.where(radars == r)[0]
        mass_centers = np.array(constants.HYDRO_CENTROIDS[r])
 
        mass_centers[:, 0] = _standardize(mass_centers[:, 0], 'Zh') 
        mass_centers[:, 1] = _standardize(mass_centers[:, 1], 'ZDR') 
        mass_centers[:, 2] = _standardize(mass_centers[:, 2], 'KDP') 
        mass_centers[:, 3] = _standardize(mass_centers[:, 3], 'RhoHV') 
        mass_centers[:, 4] = _standardize(mass_centers[:, 4], 'relH') 
    
        lapse_rate = -6.5
        relh = temp*(1000./lapse_rate)
        zh_sta = _standardize(zh[idx],'Zh')
        zdr_sta = _standardize(zdr[idx],'ZDR')
        kdp_sta = _standardize(kdp[idx],'KDP')
        rhohv_sta = _standardize(rhohv[idx],'RhoHV')
        relh_sta = _standardize(relh[idx],'relH')
        
     
        data = np.vstack((zh_sta,zdr_sta,kdp_sta,rhohv_sta,relh_sta)).T
        
        if len(data.shape) == 1:
            data = np.array([data])
    
        dist = cdist(data, mass_centers,'minkowski',p=2,w=weights)
        hydro[idx] = np.argmin(dist,axis=1)
        hydro = hydro.astype(np.int8)
    return hydro

def hydroClass_single_over_iso(radars, zh, zdr, kdp, rhohv, H_ISO0, 
                      weights = np.array([1., 1., 1., 0.75, 0.5])):
    """
    Computes the hydrometeor classes for columnar data, note that all input
    arrays except weights must have the same length
    
    Parameters
    ----------
    radars : ndarray of char
        Array of radar IDs, ('A','D','L','P','W')
    zh : ndarray
        Array of radar reflectivity in dBZ
    zdr: ndarray
        Array of diff. reflectivity in dB
    kdp: ndarray
        Array of specific phase shift on propagation in deg / km
    rhohv: ndarray
        Array of copolar correlation coefficient
    iso0_height: ndarray
        Array of 0° degree isotherm altitude
    weights: ndarray (optional)
        The weight of every input feature, zh, zdr, kdp, rhohv, temp in the
        hydrometeor classification
           
    Returns
    -------
    The hydrometeor classes as ndarray with values from 0 to 8, corresponding to
    the classes
        0 : no data
        1 : aggregates (AG)
        2 : light rain (LR)
        3 : ice crystals (CR)
        4 : rimed particles (RP)
        5 : rain (RN)
        6 : vertically aligned ice (VI)
        7 : wet snow (WS)
        8 : melting hail (MH)
        9: dry hail / high density graupel (IH/HDG)
    """
    
    from pyart.retrieve.echo_class import _standardize
    unique_radars = np.unique(radars)
    hydro = np.zeros(len(radars)) + np.nan
    
    for r in unique_radars:
        
        idx = np.where(radars == r)[0]
        mass_centers = np.array(constants.HYDRO_CENTROIDS[r])
 
        mass_centers[:, 0] = _standardize(mass_centers[:, 0], 'dBZ') 
        mass_centers[:, 1] = _standardize(mass_centers[:, 1], 'ZDR') 
        mass_centers[:, 2] = _standardize(mass_centers[:, 2], 'KDP') 
        mass_centers[:, 3] = _standardize(mass_centers[:, 3], 'RhoHV') 
        mass_centers[:, 4] = _standardize(mass_centers[:, 4], 'H_ISO0') 
    
        # lapse_rate = -6.5
        # relh = temp*(1000./lapse_rate)
        zh_sta = _standardize(zh[idx],'dBZ')
        zdr_sta = _standardize(zdr[idx],'ZDR')
        kdp_sta = _standardize(kdp[idx],'KDP')
        rhohv_sta = _standardize(rhohv[idx],'RhoHV')
        iso0_sta = _standardize(H_ISO0[idx],'H_ISO0')
        
     
        data = np.vstack((zh_sta,zdr_sta,kdp_sta,rhohv_sta,iso0_sta)).T
        
        if len(data.shape) == 1:
            data = np.array([data])
    
        dist = cdist(data, mass_centers,'minkowski',p=2,w=weights)
        hydro[idx] = np.argmin(dist,axis=1)
        hydro = hydro.astype(np.int8)
    return hydro


