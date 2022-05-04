#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 17:38:52 2019
@author: wolfensb, gugerlir
"""


import numpy as np
from scipy.spatial.distance import cdist
from textwrap import  dedent
import logging

from pyart.retrieve import kdp_leastsquare_single_window
from pyart.retrieve import hydroclass_semisupervised
from pyart.retrieve import compute_noisedBZ
from pyart.correct import smooth_phidp_single_window
from pyart.correct import calculate_attenuation_zphi
from pyart.aux_io import read_metranet

from .utils import sweepnumber_fromfile, rename_fields
from .io_data import read_status, read_vpr
from . import constants
from .lookup import get_lookup

class Radar(object):
    '''
    A class that contains polar radar data and performs some pre-processing
    before adding that data to the database or computing the QPE product
    The different elevations are stored in a dictionary rather as in a
    single pyart radar instance as this was found to be faster in practice
    '''
    def __init__(self, radname, polfiles, statusfile = None, vprfile = None):
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
        """
        
        self.sweeps = []
        self.radsweeps = {}
        
        visib = get_lookup('visibility_rad', radname)
        
    
        for f in polfiles:
            try:
                sweep = sweepnumber_fromfile(f)
                radinstance = read_metranet(f, reader = 'python')
                rename_fields(radinstance)
                visib_sweep = np.ma.array(visib[sweep].astype(np.float32), 
                                          mask = np.isnan(visib[sweep]))
                zh = radinstance.get_field(0,'ZH')
                visib_sweep = visib_sweep[0:len(zh),:]
                radinstance.add_field('VISIB',{'data': visib_sweep})
                                
                self.radsweeps[sweep] = radinstance
                self.sweeps.append(sweep)
            except:
                logging.error('Could not read file {:s}'.format(f))
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
                logging.error('Could not compute noise from status file!')
                pass
            
        if vprfile != None:
            try:
                self.vpr = read_vpr(vprfile, self.radname)
            except:
                logging.error('Could not add vpr file!')
                pass
            
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
        """
        
        for s in self.sweeps:
            radsweep = self.radsweeps[s]
            ah, pia, cor_z, _, pida, cor_zdr = calculate_attenuation_zphi(
                             radsweep,
                             refl_field='ZH',
                             zdr_field = 'ZDR',
                             phidp_field = 'PHIDP',
                             temp_field = 'T',
                             temp_ref = 'temperature',
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


