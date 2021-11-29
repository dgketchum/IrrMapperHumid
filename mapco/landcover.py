import os
from pprint import pprint
import json

from pandas import read_csv

from state_county_names_codes import state_fips_code, state_county_code, county_acres


def get_county_landcover(nass, landcover_meta=None):
    state_codes = state_fips_code()
    df = read_csv(nass, index_col='GEOID')
    dct = {}
    for s, c in state_codes.items():
        if s not in state_county_code().keys():
            continue
        for k, v in state_county_code()[s].items():
            geoid = v['GEOID']
            tot_acres = county_acres()[geoid]
            water = tot_acres['water']
            land = tot_acres['land']
            total = land + water
            if total == 0:
                pass
            try:
                irr_area = df.loc[int(geoid)]['IRR_2017']
                dry_area = df.loc[int(geoid)]['CROP_2017'] - irr_area
                irr_ratio = irr_area / total
                dry_ratio = dry_area / total
                uncult_ratio = (land - irr_area - dry_area) / total
                water_ratio = water / total
                d = {'dryland': dry_ratio, 'irrigated': irr_ratio, 'uncultivated': uncult_ratio,
                     'water': water_ratio}
                if any([np.isnan(x) for x in [dry_ratio, irr_ratio, uncult_ratio, water_ratio]]):
                    dct[geoid] = {'cover': 'unknown'}
                else:
                    dct[geoid] = d
            except KeyError:
                dct[geoid] = {'cover': 'unknown'}
    if landcover_meta:
        with open(landcover_meta, 'w') as fp:
            fp.write(json.dumps(dct, indent=4, sort_keys=True))
        return None
    pprint(dct)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
