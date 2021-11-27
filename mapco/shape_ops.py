import os
import sys
from subprocess import check_call
from collections import OrderedDict

import numpy as np
import fiona
from shapely.geometry import shape
from pandas import read_csv

from state_county_names_codes import state_fips_code, state_county_code
from cdl import cdl_crops
from climate import generate_precip_records, get_prec_anomaly

sys.path.append('/home/dgketchum/PycharmProjects/EEMapper')
from map import distribute_points as dstpts

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
        if s != 'NE':
            continue
        s_dir = os.path.join(out_dir, s)
        if not os.path.isdir(s_dir):
            os.mkdir(s_dir)
        state_fields = os.path.join(fields_dir, '{}.shp'.format(s))
        for k, v in state_county_code()[s].items():
            if v['GEOID'] != '31155':
                continue
            co_field_filename = '{}.shp'.format(v['GEOID'])
            clip_src = os.path.join(county_shp_dir, co_field_filename)
            county_fields = os.path.join(out_dir, s, co_field_filename)
            cmd = [OGR, '-clipsrc', clip_src, '-s_srs', 'EPSG:4326', '-t_srs', AEA, county_fields, state_fields]
            check_call(cmd)


def attribute_county_fields(county_shp_dir, out_co_dir, min_area=5e4):
    state_codes = state_fips_code()
    for s, c in state_codes.items():
        if s != 'NE':
            continue
        s_dir = os.path.join(out_co_dir, s)
        if not os.path.isdir(s_dir):
            os.mkdir(s_dir)

        for k, v in state_county_code()[s].items():
            geoid = v['GEOID']
            if geoid != '31155':
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
                        barea = geo.envelope.area / geo.area
                        props = feat['properties']
                        cdl_keys = [x for x in props.keys() if 'CROP_' in x]
                        cdl_ = [props[x] for x in cdl_keys]
                        vals, counts = np.unique(cdl_, return_counts=True)
                        cdl_mode = vals[np.argmax(counts)].item()
                        feat['properties'] = {k: v for k, v in feat['properties'].items() if k in cdl_keys}
                        feat['properties'].update({'FID': ct,
                                                   'cdl_mode': cdl_mode,
                                                   'popper': popper_,
                                                   'bratio': barea})

                        features.append(feat)
                        ct += 1
                    except TypeError:
                        bad_geo_ct += 1
            new_attrs = [('FID', 'int:9'), ('cdl_mode', 'int:9'),
                         ('popper', 'float'),
                         ('bratio', 'float')] + [(x, 'int') for x in cdl_keys]
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


def write_potential_training_data(fields, training_dir, climate_dir, max_count=500):
    crops = cdl_crops().keys()
    state_codes = state_fips_code()
    for s, c in state_codes.items():
        if s != 'NE':
            continue
        s_dir = os.path.join(training_dir, s)
        if not os.path.isdir(s_dir):
            os.mkdir(s_dir)

        for k, v in state_county_code()[s].items():
            geoid = v['GEOID']
            if geoid != '31155':
                continue

            dry_years = get_dry_years(geoid, climate_dir, n_years=3)
            p_ct, d_ct, u_ct = 0, 0, 0
            raw_fields = os.path.join(fields, s, '{}.shp'.format(geoid))
            pivot, uncult, dry = [], [], []
            with fiona.open(raw_fields, 'r') as src:
                meta = src.meta
                meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
                    [('FID', 'int:9')]), 'geometry': 'Polygon'}

                for f in src:
                    props = f['properties']
                    popper_ = props['popper']
                    bratio = props['bratio']
                    cdl_ = props['CROP_{}'.format(dry_years[0])]
                    if cdl_ == 61:
                        if len(dry) < max_count:
                            dry.append(f)
                            d_ct += 1
                    elif popper_ > 0.95:
                        if len(pivot) < max_count:
                            pivot.append(f)
                            p_ct += 1
                    elif 0.7 < popper_ < 0.75 and cdl_ in crops and bratio < 1.15:
                        if len(dry) < max_count:
                            dry.append(f)
                            d_ct += 1
                    elif cdl_ not in crops:
                        if len(uncult) < max_count:
                            uncult.append(f)
                            u_ct += 1

            for features, prepend in zip([pivot, uncult, dry], ['p', 'u', 'd']):
                out_shp = os.path.join(s_dir, '{}_{}.shp'.format(prepend, geoid))
                ct = 1
                with fiona.open(out_shp, 'w', **meta) as dst:
                    for feat in features:
                        f = {'type': 'Feature', 'properties': {'FID': ct}, 'geometry': feat['geometry']}
                        dst.write(f)
            print('{} {} dry years: {}, {} pivot, {} dry, {} uncult'.format(s, geoid, dry_years, p_ct, d_ct, u_ct))


def distribute_points(training_dir, n_points, nass_data):
    df = read_csv(nass_data)
    state_codes = state_fips_code()
    for s, c in state_codes.items():
        if s != 'NE':
            continue
        s_dir = os.path.join(training_dir, s)
        if not os.path.isdir(s_dir):
            os.mkdir(s_dir)

        for k, v in state_county_code()[s].items():
            geoid = v['GEOID']
            if geoid != '31155':
                continue

            kwargs = {'irrigated_path': os.path.join(training_dir, 'p_{}.shp'.format(geoid)),
                      'uncultivated_path': os.path.join(training_dir, 'u_{}.shp'.format(geoid)),
                      'unirrigated_path': os.path.join(training_dir, 'd_{}.shp'.format(geoid)),
                      'wetland_path': os.path.join(training_dir, 'wetlands_9NOV2021.shp')}

            kwargs.update({
                'irrigated': 100,
                'wetland': 100,
                'uncultivated': 100,
            })


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
    state_fields_ = os.path.join(gis, 'openET/OpenET_GeoDatabase')
    co_fields_ = os.path.join(gis, 'openET/county_fields_aea')
    co_fields_attr = os.path.join(gis, 'openET/county_fields_attr')
    # clip_field_boundaries_county(state_fields_, co_shp_, co_fields_)
    # attribute_county_fields(co_fields_, co_fields_attr)
    co_climate = os.path.join(gis, 'training_data', 'humid', 'county_precip_normals')
    potential = os.path.join(gis, 'training_data', 'humid', 'potential')
    # write_potential_training_data(co_fields_attr, potential, co_climate)
    nass = os.path.join(gis, 'nass_data', 'nass_irr_crop_new.csv')
    distribute_points(potential, n_points=4000, nass_data=nass)
# ========================= EOF ====================================================================
