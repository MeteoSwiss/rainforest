import math


class GPSConverter(object):
    '''
    GPS Converter class which is able to perform convertions between the 
    CH1903 and WGS84 system.
    '''
    def CHtoWGSheight(self, y, x, h):
        """Convert CH y/x/h to WGS height"""
        # Auxiliary values (% Bern)
        y_aux = (y-600000) / 1000000
        x_aux = (x-200000) / 1000000
        h = h + 49.55 - 12.60*y_aux - 22.64*x_aux
        return h

    def CHtoWGSlat(self, y, x):
        """Convert CH y/x to WGS lat"""
        # Auxiliary values (% Bern)
        y_aux = (y-600000) / 1000000
        x_aux = (x-200000) / 1000000
        lat = (16.9023892
               + 3.238272 * x_aux
               - 0.270978 * y_aux**2
               - 0.002528 * x_aux**2
               - 0.0447   * y_aux**2 * x_aux
               - 0.0140   * x_aux**3)
        # Unit 10000" to 1" and convert seconds to degrees (dec)
        lat = lat * 10000 / 3600
        return lat

    def CHtoWGSlng(self, y, x):
        """Convert CH y/x to WGS long"""
        # Auxiliary values (% Bern)
        y_aux = (y-600000) / 1000000
        x_aux = (x-200000) / 1000000
        lng = (2.6779094
               + 4.728982 * y_aux
               + 0.791484 * y_aux * x_aux
               + 0.1306   * y_aux * x_aux**2
               - 0.0436   * y_aux**3)
        # Unit 10000" to 1" and convert seconds to degrees (dec)
        lng = lng * 10000 / 3600
        return lng

    def WGStoCHh(self, lat, lng, h):
        """Convert WGS lat/long (° dec) and height to CH h"""
        # Decimal degrees to seconds
        lat = lat * 3600
        lng = lng * 3600
        # Auxiliary values (% Bern)
        lat_aux = (lat-169028.66) / 10000
        lng_aux = (lng-26782.5) / 10000
        h = h - 49.55 + 2.73*lng_aux + 6.94*lat_aux
        return h

    def WGStoCHx(self, lat, lng):
        """Convert WGS lat/long (° dec) to CH x"""
        # Decimal degrees to seconds
        lat = lat * 3600
        lng = lng * 3600
        # Auxiliary values (% Bern)
        lat_aux = (lat-169028.66) / 10000
        lng_aux = (lng-26782.5) / 10000
        x = (200147.07
             + 308807.95 * lat_aux
             +   3745.25 * lng_aux**2
             +     76.63 * lat_aux**2
             -    194.56 * lng_aux**2 * lat_aux
             +    119.79 * lat_aux**3)
        return x

    def WGStoCHy(self, lat, lng):
        """Convert WGS lat/long (° dec) to CH y"""
        # Decimal degrees to seconds
        lat = lat * 3600
        lng = lng * 3600
        # Auxiliary values (% Bern)
        lat_aux = (lat-169028.66) / 10000
        lng_aux = (lng-26782.5) / 10000
        y = (600072.37
             + 211455.93 * lng_aux
             -  10938.51 * lng_aux * lat_aux
             -      0.36 * lng_aux * lat_aux**2
             -     44.54 * lng_aux**3)
        return y

    def LV03toWGS84(self, east, north, height):
        '''
        Convert LV03 to WGS84. Return a tuple of floating point numbers
        containing lat, long, and height
        '''
        return (self.CHtoWGSlat(east, north),
                self.CHtoWGSlng(east, north),
                self.CHtoWGSheight(east, north, height))
        
    def WGS84toLV03(self, latitude, longitude, ellHeight):
        '''
        Convert WGS84 to LV03. Return a tuple of floating point numbers
        containing east, north, and height
        '''
        return (self.WGStoCHy(latitude, longitude),
                self.WGStoCHx(latitude, longitude),
                self.WGStoCHh(latitude, longitude, ellHeight))

