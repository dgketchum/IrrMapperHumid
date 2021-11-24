import os
from pprint import pprint
from subprocess import check_call
from collections import OrderedDict

import fiona
from rasterstats import zonal_stats
from shapely.geometry import shape

from state_county_names_codes import state_fips_code, state_county_code
from cdl import cdl_crops

OGR = '/usr/bin/ogr2ogr'
AEA = '+proj=aea +lat_0=40 +lon_0=-96 +lat_1=20 +lat_2=60 +x_0=0 +y_0=0 +ellps=GRS80 ' \
      '+towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
WGS = '+proj=longlat +datum=WGS84 +no_defs'


def clip_field_boundaries_county(fields_dir, county_shp_dir, out_dir):
    """clip field boundaries from state to county, project to albers equal area EPSG:102008"""
    state_codes = state_fips_code()
    for s, c in state_codes.items():
        if s != 'SD':
            continue
        s_dir = os.path.join(out_dir, s)
        if not os.path.isdir(s_dir):
            os.mkdir(s_dir)
        state_fields = os.path.join(fields_dir, '{}.shp'.format(s))
        for k, v in state_county_code()[s].items():
            if v['GEOID'] != '46125':
                continue
            co_field_filename = '{}.shp'.format(v['GEOID'])
            clip_src = os.path.join(county_shp_dir, co_field_filename)
            county_fields = os.path.join(out_dir, s, co_field_filename)
            cmd = [OGR, '-clipsrc', clip_src, '-s_srs', 'EPSG:4326', '-t_srs', AEA, county_fields, state_fields]
            check_call(cmd)


def attribute_county_fields(county_shp_dir, out_co_dir, cdl):
    state_codes = state_fips_code()
    for s, c in state_codes.items():
        if s != 'SD':
            continue
        s_dir = os.path.join(out_co_dir, s)
        if not os.path.isdir(s_dir):
            os.mkdir(s_dir)

        for k, v in state_county_code()[s].items():
            geoid = v['GEOID']
            if geoid != '46125':
                continue
            blank_fields = os.path.join(county_shp_dir, s, '{}.shp'.format(geoid))
            attr_fields = os.path.join(out_co_dir, s, '{}.shp'.format(geoid))

            ct = 1
            geo = []
            bad_geo_ct = 0
            with fiona.open(blank_fields) as src:
                meta = src.meta
                for feat in src:
                    try:
                        _ = feat['geometry']['type']
                        geo.append(feat)
                    except TypeError:
                        bad_geo_ct += 1

            input_feats = len(geo)
            print('{} features in {}'.format(input_feats, blank_fields))
            temp_file = attr_fields.replace('.shp', '_temp.shp')
            with fiona.open(temp_file, 'w', **meta) as tmp:
                for feat in geo:
                    tmp.write(feat)

            meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
                [('FID', 'int:9'), ('CDL', 'int:9'), ('popper', 'float')]),
                              'geometry': 'Polygon'}
            in_raster = os.path.join(cdl, 'CDL_2017_{}.tif'.format(s))
            stats = zonal_stats(temp_file, in_raster, stats=['majority'], nodata=0.0, categorical=True)

            include_codes = [k for k in cdl_crops().keys()]

            ct_inval = 0
            ct_non_crop = 0
            with fiona.open(attr_fields, mode='w', **meta) as out:
                for attr, g in zip(stats, geo):

                    try:
                        cdl_code = int(attr['majority'])
                    except TypeError:
                        cdl_code = 0

                    cropped = attr['majority'] in include_codes

                    feat = {'type': 'Feature',
                            'properties': {'FID': ct,
                                           'CDL': cdl_code},
                            'geometry': g['geometry']}

                    if not feat['geometry']:
                        ct_inval += 1
                    elif not shape(feat['geometry']).is_valid:
                        ct_inval += 1
                    else:
                        out.write(feat)
                        ct += 1
                        ct_non_crop += 1

                print('{} in, {} out, {} invalid, {}'.format(input_feats, ct - 1, ct_inval, attr_fields))
                rm_files = [temp_file.replace('shp', x) for x in ['prj', 'cpg', 'dbf', 'shx']] + [temp_file]
                [os.remove(f) for f in rm_files]


if __name__ == '__main__':
    gis = os.path.join('/media/research', 'IrrigationGIS')
    if not os.path.exists(gis):
        gis = '/home/dgketchum/data/IrrigationGIS'
    co_shp_ = os.path.join(gis, 'boundaries/counties/county_shapefiles')
    cdl_ = os.path.join(gis, 'cdl', 'wgs')
    state_fields_ = os.path.join(gis, 'openET/OpenET_GeoDatabase')
    co_fields_ = os.path.join(gis, 'openET/county_fields_blank_aea')
    co_fields_attr = os.path.join(gis, 'openET/county_fields_attr')
    # clip_field_boundaries_county(state_fields_, co_shp_, co_fields_)
    attribute_county_fields(co_fields_, co_fields_attr, cdl_)
# ========================= EOF ====================================================================
