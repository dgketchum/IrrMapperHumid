import os
from pprint import pprint
import fiona

from state_county_names_codes import state_fips_code


def write_state_county_data(shp, ):
    st_fp = state_fips_code()
    codes = {v: k for k, v in st_fp.items()}
    with fiona.open(shp, 'r') as src:
        dct = {}
        for f in src:
            props = f['properties']
            st = codes[props['STATEFP']]
            if st not in dct.keys():
                dct[st] = {props['COUNTYFP']: {'NAME': props['NAME'],
                                               'GEOID': props['GEOID']}}
                continue
            dct[st][props['COUNTYFP']] = {'NAME': props['NAME'],
                                          'GEOID': props['GEOID']}
    pprint(dct)


if __name__ == '__main__':
    gis = os.path.join('/media/research', 'IrrigationGIS')
    if not os.path.exists(gis):
        gis = '/home/dgketchum/data/IrrigationGIS'
    county_shp = '/media/research/IrrigationGIS/boundaries/counties/cb_2017_us_county_20m_wgs.shp'
    # county_shp = '/media/research/IrrigationGIS/boundaries/us_states_tiger_wgs.shp'
    write_state_county_data(county_shp)
# ========================= EOF ====================================================================
