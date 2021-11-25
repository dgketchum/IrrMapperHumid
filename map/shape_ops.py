import os
from pprint import pprint
from subprocess import check_call
from collections import OrderedDict

import numpy as np
import fiona
from rasterstats import zonal_stats
from shapely.geometry import shape, Polygon

from state_county_names_codes import state_fips_code, state_county_code
from cdl import cdl_crops
from climate import generate_precip_records, get_prec_anomaly

OGR = '/usr/bin/ogr2ogr'
AEA = '+proj=aea +lat_0=40 +lon_0=-96 +lat_1=20 +lat_2=60 +x_0=0 +y_0=0 +ellps=GRS80 ' \
      '+towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
WGS = '+proj=longlat +datum=WGS84 +no_defs'


def popper(geometry):
    p = (4 * np.pi * geometry.area) / (geometry.boundary.length ** 2.)
    return p


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


def attribute_county_fields(county_shp_dir, out_co_dir, min_area=5e4):
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
            raw_fields = os.path.join(county_shp_dir, s, '{}.shp'.format(geoid))
            attr_fields = os.path.join(out_co_dir, s, '{}.shp'.format(geoid))

            fct, ct, non_polygon, bad_geo_ct = 0, 0, 0, 0
            features = []
            with fiona.open(raw_fields) as src:
                meta = src.meta
                for feat in src:
                    fct += 1
                    try:
                        gtype = feat['geometry']['type']
                        if gtype != 'Polygon':
                            non_polygon += 1
                            continue
                        geo = shape(feat['geometry'])
                        if geo.area < min_area:
                            continue
                        popper_ = float(popper(geo))
                        props = feat['properties']
                        cdl_keys = [x for x in props.keys() if 'CROP_' in x]
                        cdl_ = [props[x] for x in cdl_keys]
                        vals, counts = np.unique(cdl_, return_counts=True)
                        cdl_mode = vals[np.argmax(counts)].item()
                        feat['properties'] = {k: v for k, v in feat['properties'].items() if k in cdl_keys}
                        feat['properties'].update({'FID': ct,
                                                   'cdl_mode': cdl_mode,
                                                   'popper': popper_})

                        features.append(feat)
                        ct += 1
                    except TypeError:
                        bad_geo_ct += 1
            new_attrs = [('FID', 'int:9'), ('cdl_mode', 'int:9'), ('popper', 'float')] + [(x, 'int') for x in cdl_keys]
            meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(new_attrs), 'geometry': 'Polygon'}

            ct_inval, wct = 0, 0
            with fiona.open(attr_fields, mode='w', **meta) as out:
                for feat in features:

                    if not feat['geometry']:
                        ct_inval += 1
                    elif not shape(feat['geometry']).is_valid:
                        ct_inval += 1
                    else:
                        out.write(feat)
                        wct += 1

                print('{} in, {} written, {} invalid, {}'.format(fct, wct, ct_inval, attr_fields))


def write_potential_training_data(fields, training_dir, climate_dir):
    crops = cdl_crops().keys()
    state_codes = state_fips_code()
    for s, c in state_codes.items():
        if s != 'SD':
            continue
        s_dir = os.path.join(fields, s)
        if not os.path.isdir(s_dir):
            os.mkdir(s_dir)

        for k, v in state_county_code()[s].items():
            geoid = v['GEOID']
            if geoid != '46125':
                continue

            dry_years = get_dry_years(geoid, climate_dir, n_years=3)

            raw_fields = os.path.join(fields, s, '{}.shp'.format(geoid))
            for year in dry_years:
                pivot, uncult, dry = [], [], []
                with fiona.open(raw_fields, 'r') as src:
                    meta = src.meta
                    meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
                        [('FID', 'int:9')]), 'geometry': 'Polygon'}

                    for f in src:
                        props = f['properties']
                        popper_ = props['popper']
                        cdl_ = props['CROP_{}'.format(year)]
                        if popper_ > 0.9:
                            pivot.append(f)
                        elif 0.7 < popper_ < 0.8 and cdl_ in crops:
                            dry.append(f)
                        elif cdl_ not in crops:
                            uncult.append(f)

                for features, t_dir in zip([pivot, uncult, dry], ['pivot', 'uncultivated', 'unirrigated']):
                    out_shp = os.path.join(training_dir, t_dir, '{}_{}.shp'.format(geoid, year))
                    ct = 1
                    with fiona.open(out_shp, 'w', **meta) as dst:
                        for feat in features:
                            f = {'type': 'Feature', 'properties': {'FID': ct}, 'geometry': feat['geometry']}
                            dst.write(f)


def get_dry_years(geoid, climate_dir, n_years=3):
    precip_record = os.path.join(climate_dir, '{}.csv'.format(geoid))
    if not os.path.exists(precip_record):
        generate_precip_records(climate_dir, geoid)
    precip_record = get_prec_anomaly(precip_record)
    dry_years = [x[0] for x in precip_record[:n_years]]
    return dry_years


if __name__ == '__main__':
    gis = os.path.join('/media/research', 'IrrigationGIS')
    if not os.path.exists(gis):
        gis = '/home/dgketchum/data/IrrigationGIS'
    co_shp_ = os.path.join(gis, 'boundaries/counties/county_shapefiles')
    # cdl_ = os.path.join(gis, 'cdl', 'wgs')
    state_fields_ = os.path.join(gis, 'openET/OpenET_GeoDatabase')
    co_fields_ = os.path.join(gis, 'openET/county_fields_aea')
    co_fields_attr = os.path.join(gis, 'openET/county_fields_attr')
    # clip_field_boundaries_county(state_fields_, co_shp_, co_fields_)
    # attribute_county_fields(co_fields_, co_fields_attr)
    co_climate = os.path.join(gis, 'training_data', 'humid', 'county_precip_normals')
    potential = os.path.join(gis, 'training_data', 'humid', 'potential')
    write_potential_training_data(co_fields_attr, potential, co_climate)
# ========================= EOF ====================================================================
