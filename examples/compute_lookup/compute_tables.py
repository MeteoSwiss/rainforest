from rainforest.common.lookup import calc_lookup

RADARS = ["A", "D", "L", "P", "W"]
LOOKUP = ["cartcoords_radL", "cartcoords_radH", "station_to_qpegrid"]
for rad in RADARS:
    for lut in LOOKUP:
        print(f"Calculating lut {lut} for rad {rad}")
        calc_lookup(lut, rad)
