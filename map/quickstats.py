import os
import json
from copy import deepcopy

import numpy as np
from pandas import concat, read_csv, read_table, DataFrame

DROP = ['SOURCE_DESC', 'SECTOR_DESC', 'GROUP_DESC',
        'COMMODITY_DESC', 'CLASS_DESC', 'PRODN_PRACTICE_DESC',
        'UTIL_PRACTICE_DESC', 'STATISTICCAT_DESC', 'UNIT_DESC',
        'SHORT_DESC', 'DOMAIN_DESC', 'DOMAINCAT_DESC', 'STATE_FIPS_CODE',
        'ASD_CODE', 'ASD_DESC', 'COUNTY_ANSI',
        'REGION_DESC', 'ZIP_5', 'WATERSHED_CODE',
        'WATERSHED_DESC', 'CONGR_DISTRICT_CODE', 'COUNTRY_CODE',
        'COUNTRY_NAME', 'LOCATION_DESC', 'YEAR', 'FREQ_DESC',
        'BEGIN_CODE', 'END_CODE', 'REFERENCE_PERIOD_DESC',
        'WEEK_ENDING', 'LOAD_TIME', 'VALUE', 'AGG_LEVEL_DESC',
        'CV_%', 'STATE_ALPHA', 'STATE_NAME', 'COUNTY_NAME']


def get_nass(csv, out_file, old_nass=None):
    first = True
    if old_nass:
        old_df = read_csv(old_nass)
        old_df.index = old_df['FIPS']
    for c in csv:
        print(c)
        try:
            df = read_table(c, sep='\t')
            assert len(list(df.columns)) > 2
        except AssertionError:
            df = read_csv(c)
        df.dropna(axis=0, subset=['COUNTY_CODE'], inplace=True, how='any')
        df['GEOID'] = df.apply(lambda row: '{}{}'.format(str(row.loc['STATE_FIPS_CODE']).rjust(2, '0'),
                                                         str(int(row.loc['COUNTY_CODE'])).rjust(3, '0')), axis=1)
        df.index = df['GEOID']
        cdf = deepcopy(df)
        cdf['ST_CNTY_STR'] = cdf['STATE_ALPHA'] + '_' + cdf['COUNTY_NAME']
        cdf = cdf[(cdf['SOURCE_DESC'] == 'CENSUS') &
                  (cdf['COMMODITY_DESC'] == 'AG LAND') &
                  (cdf['STATISTICCAT_DESC'] == 'AREA') &
                  (cdf['UNIT_DESC'] == 'ACRES') &
                  (df['SHORT_DESC'] == 'AG LAND, CROPLAND - ACRES') &
                  (cdf['DOMAIN_DESC'] == 'TOTAL')]
        cdf['VALUE'] = cdf['VALUE'].map(lambda x: np.nan if 'D' in x else int(x.replace(',', '')))

        df['ST_CNTY_STR'] = df['STATE_ALPHA'] + '_' + df['COUNTY_NAME']
        df = df[(df['SOURCE_DESC'] == 'CENSUS') &
                (df['SECTOR_DESC'] == 'ECONOMICS') &
                (df['GROUP_DESC'] == 'FARMS & LAND & ASSETS') &
                (df['COMMODITY_DESC'] == 'AG LAND') &
                (df['CLASS_DESC'] == 'ALL CLASSES') &
                (df['PRODN_PRACTICE_DESC'] == 'IRRIGATED') &
                (df['UTIL_PRACTICE_DESC'] == 'ALL UTILIZATION PRACTICES') &
                (df['STATISTICCAT_DESC'] == 'AREA') &
                (df['UNIT_DESC'] == 'ACRES') &
                (df['SHORT_DESC'] == 'AG LAND, IRRIGATED - ACRES') &
                (df['DOMAIN_DESC'] == 'TOTAL')]
        df['VALUE'] = df['VALUE'].map(lambda x: np.nan if 'D' in x else int(x.replace(',', '')))
        if first:
            first = False
            ndf = df[['YEAR', 'GEOID']]
            ndf['IRR_VALUE_{}'.format(df.iloc[0]['YEAR'])] = df['VALUE']
            ndf['CROP_VALUE_{}'.format(df.iloc[0]['YEAR'])] = cdf['VALUE']
        else:
            ndf['IRR_{}'.format(df.iloc[0]['YEAR'])] = df['VALUE']
            ndf['CROP_{}'.format(df.iloc[0]['YEAR'])] = cdf['VALUE']

    ndf.to_csv(out_file.replace('.csv', '_new.csv'))
    df = concat([old_df, ndf], axis=1)
    df.to_csv(out_file)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    nass_tables = os.path.join(root, 'nass_data')
    irr_tables = os.path.join(nass_tables, 'counties_v2', 'noCdlMask_minYr5')

    old_data = os.path.join(nass_tables, 'old_nass.csv')
    old_data_dir = os.path.join(nass_tables, 'ICPSR_35206')

    _files = [os.path.join(nass_tables, x) for x in ['qs.census2002.txt',
                                                     'qs.census2007.txt',
                                                     'qs.census2012.txt',
                                                     'qs.census2017.txt']]
    merged = os.path.join(nass_tables, 'nass_irr_crop.csv')
    get_nass(_files, merged, old_nass=False)

# ========================= EOF ====================================================================
