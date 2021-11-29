import os
import sys
import json
from pprint import pprint
from subprocess import check_call, run, PIPE
from collections import OrderedDict

import numpy as np
import fiona
from shapely.geometry import shape

from mapco.state_county_names_codes import state_fips_code, state_county_code
from mapco.cdl import cdl_crops
from mapco.climate import get_climate

sys.path.append('/home/dgketchum/PycharmProjects/EEMapper')
from map import distribute_points as dstpts

home = os.path.expanduser('~')
conda = os.path.join(home, 'miniconda3', 'envs')
if not os.path.exists(conda):
    conda = conda.replace('miniconda3', 'miniconda')
EE = os.path.join(conda, 'metric', 'bin', 'earthengine')
GS = os.path.join(conda, 'metric', 'bin', 'gsutil')
OGR = '/usr/bin/ogr2ogr'
AEA = '+proj=aea +lat_0=40 +lon_0=-96 +lat_1=20 +lat_2=60 +x_0=0 +y_0=0 +ellps=GRS80 ' \
      '+towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
WGS = '+proj=longlat +datum=WGS84 +no_defs'

ne_test = ['31097', '31035', '31011', '31181', '31059', '31127', '31001', '31053', '31027', '31071', '31151',
           '31185', '31067', '31051', '31133', '31125', '31079', '31147', '31179', '31173', '31081', '31095',
           '31003', '31119', '31089', '31093', '31099', '31023', '31129', '31153', '31019', '31155', '31139',
           '31141', '31167', '31107', '31039', '31021', '31169', '31037', '31015', '31109', '31055', '31175',
           '31143', '31131', '31061', '31183', '31121', '31043', '31177', '31163', '31159', '31077']

TEST_COUNTIES = [c for c in ne_test if c not in ['31081', '31109', '31121', '31167', '31169', '31173']]


def county_training_devlopment(state_fields, training_dir, county_shape_wgs,
                               county_fields_aea, county_attr,
                               county_shape_aea, climate, landcover):
    cmd = [GS, 'ls', os.path.join(_bucket, 'county_points')]
    existing = [x.decode('utf-8') for x in run(cmd, stdout=PIPE).stdout.splitlines()]
    failures = []
    state_codes = state_fips_code()
    for s, c in state_codes.items():
        if s not in state_county_code().keys():
            continue
        state_field_bounds = os.path.join(state_fields, '{}.shp'.format(s))
        for k, v in state_county_code()[s].items():
            geoid = v['GEOID']
            if geoid not in TEST_COUNTIES:
                continue

            bucket_file = os.path.join(_bucket, 'county_points', 'pts_{}.shp'.format(geoid))
            if bucket_file in existing:
                print('skip ', geoid)
                continue
            try:
                clip_field_boundaries_county(geoid, state_field_bounds, county_shape_wgs, county_fields_aea)
                attribute_county_fields(geoid, county_fields_aea, county_attr)
                write_potential_training_data(geoid, county_attr, training_dir, climate)
                distribute_points(geoid, training_dir, n_points=4000, boundary_dir=county_shape_aea,
                                  landcover_meta=landcover, climate=climate, overwrite=False)
                push_points_to_asset(geoid, training_dir, bucket=_bucket, overwrite=False)
            except Exception as e:
                print(geoid, e)
                failures.append(geoid)
    print(failures)


def points_to_geographic(geoid, _dir):
    target_dir = os.path.join(_dir, geoid)
    in_shp = os.path.join(target_dir, 'pts_{}_aea.shp'.format(geoid))
    out_shp = os.path.join(in_shp.replace('_aea', ''))
    cmd = [OGR, '-f', 'ESRI Shapefile', '-t_srs', WGS, '-s_srs', AEA, out_shp, in_shp]
    check_call(cmd)
    remove = [os.path.join(target_dir, x) for x in os.listdir(target_dir) if '_aea' in x and geoid in x]
    [os.remove(r) for r in remove]
    print(out_shp)


def popper(geometry):
    p = (4 * np.pi * geometry.area) / (geometry.boundary.length ** 2.)
    return p


def clip_field_boundaries_county(geoid, state_fields, county_shp_dir, out_dir, overwrite=False):
    """clip field boundaries from state to county, project to albers equal area EPSG:102008"""
    co_field_filename = '{}.shp'.format(geoid)
    clip_src = os.path.join(county_shp_dir, co_field_filename)
    county_fields = os.path.join(out_dir, co_field_filename)
    if os.path.exists(county_fields) and not overwrite:
        print('{} exists, skipping'.format(county_fields))
    cmd = [OGR, '-clipsrc', clip_src, '-s_srs', 'EPSG:4326', '-t_srs', AEA, county_fields, state_fields]
    check_call(cmd)


def attribute_county_fields(geoid, county_shp_dir, out_co_dir, min_area=5e4, overwrite=False):
    raw_fields = os.path.join(county_shp_dir, '{}.shp'.format(geoid))
    attr_fields = os.path.join(out_co_dir, '{}.shp'.format(geoid))
    if os.path.exists(attr_fields) and not overwrite:
        print('{} exists, skipping'.format(attr_fields))
        return None
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
    if fct == 0:
        raise AttributeError('{} has no fields'.format(raw_fields))
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


def write_potential_training_data(geoid, fields, training_dir, climate_dir, max_count=500, overwrite=False):
    crops = cdl_crops().keys()
    c_dir = os.path.join(training_dir, geoid)
    if not os.path.isdir(c_dir):
        os.mkdir(c_dir)
    if os.path.exists(os.path.join(c_dir, 'd_{}.shp'.format(geoid))) and not overwrite:
        print('{} exists, skipping'.format('d_{}.shp'.format(geoid)))
    dry_years = get_climate(geoid, climate_dir)
    p_ct, d_ct, u_ct = 0, 0, 0
    raw_fields = os.path.join(fields, '{}.shp'.format(geoid))
    pivot, uncult, dry = [], [], []
    with fiona.open(raw_fields, 'r') as src:
        meta = src.meta

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
                    f['properties']['YEAR'] = dry_years[0]
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
        out_shp = os.path.join(c_dir, '{}_{}.shp'.format(prepend, geoid))
        ct = 1
        if prepend == 'p' and len(features) < 1:
            continue
        meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
            [('FID', 'int:9')]), 'geometry': 'Polygon'}
        with fiona.open(out_shp, 'w', **meta) as dst:
            for feat in features:
                f = {'type': 'Feature', 'properties': {'FID': ct}, 'geometry': feat['geometry']}
                dst.write(f)
    print('{} dry years: {}, {} pivot, {} dry, {} uncult'.format(geoid, dry_years, p_ct, d_ct, u_ct))


def distribute_points(geoid, training_dir, n_points, boundary_dir, landcover_meta, climate, overwrite=False):
    c_dir = os.path.join(training_dir, geoid)
    eventual_name = os.path.join(c_dir, 'pts_{}.shp'.format(geoid))

    if os.path.exists(eventual_name) and not overwrite:
        print('{} exists, skipping'.format(eventual_name))
        return None

    out_name = os.path.join(c_dir, 'pts_{}_aea.shp'.format(geoid))
    with open(landcover_meta, 'r') as fp:
        dct = json.load(fp)[geoid]
    if 'cover' in dct.keys():
        print('{} landcover stats unknown, skipping'.format(geoid))

    intersect_shape = os.path.join(boundary_dir, '{}.shp'.format(geoid))
    dry_years = get_climate(geoid, climate, return_dry=True, n_years=3)

    wetlands = '/home/dgketchum/data/IrrigationGIS/compiled_training_data/aea/wetlands_22NOV2021.shp'
    if not os.path.exists(wetlands):
        wetlands = '/media/research/IrrigationGIS/compiled_training_data/aea/wetlands_22NOV2021.shp'
    kwargs = {
        'irrigated_path': os.path.join(c_dir, 'p_{}.shp'.format(geoid)),
        'uncultivated_path': os.path.join(c_dir, 'u_{}.shp'.format(geoid)),
        'unirrigated_path': os.path.join(c_dir, 'd_{}.shp'.format(geoid)),
        'wetland_path': wetlands,
    }
    pprint({k[0]: '{:.2f}'.format(v) for k, v in dct.items()})

    kwargs.update({
        'irrigated': int(dct['irrigated'] * n_points),
        'wetland': int(dct['water'] * n_points),
        'uncultivated': int(dct['uncultivated'] * n_points),
        'unirrigated': int(dct['dryland'] * n_points),
        'intersect': intersect_shape,
        'intersect_buffer': 100000,
        'years': dry_years})

    prs = dstpts.PointsRunspec(buffer=-20, **kwargs)
    prs.save_sample_points(out_name)

    points_to_geographic(geoid, training_dir)


def push_points_to_asset(geoid, training_dir, bucket, overwrite=False):
    c_dir = os.path.join(training_dir, geoid)
    local_files = [os.path.join(c_dir, 'pts_{}.{}'.format(geoid, ext)) for ext in
                   ['shp', 'prj', 'shx', 'dbf']]
    bucket = os.path.join(bucket, 'county_points')
    bucket_files = [os.path.join(bucket, 'pts_{}.{}'.format(geoid, ext)) for ext in
                    ['shp', 'prj', 'shx', 'dbf']]
    cmd = [GS, 'ls', bucket]
    existing = [x.decode('utf-8') for x in run(cmd, stdout=PIPE).stdout.splitlines()]
    if bucket_files[0] in existing and not overwrite:
        print('{} exists, skipping'.format(bucket_files[0]))

    for lf, bf in zip(local_files, bucket_files):
        cmd = [GS, 'cp', lf, bf]
        check_call(cmd)

    asset_id = os.path.basename(bucket_files[0]).split('.')[0]
    ee_dst = 'users/dgketchum/points/county/{}'.format(asset_id)
    cmd = [EE, 'upload', 'table', '-f', '--asset_id={}'.format(ee_dst), bucket_files[0]]
    check_call(cmd)
    print(asset_id, bucket_files[0])


if __name__ == '__main__':
    gis = os.path.join('/media/research', 'IrrigationGIS')
    if not os.path.exists(gis):
        gis = '/home/dgketchum/data/IrrigationGIS'
    _bucket = 'gs://wudr'
    _state_fields = os.path.join(gis, 'openET/OpenET_GeoDatabase')
    _training_dir = os.path.join(gis, 'training_data', 'humid', 'potential')
    _co_shapes_wgs = os.path.join(gis, 'boundaries/counties/county_shapefiles_wgs')
    _co_fields_wgs = os.path.join(gis, 'openET/county_fields_aea')
    _co_fields_attr = os.path.join(gis, 'openET/county_fields_attr')
    _co_shapes_aea = os.path.join(gis, 'boundaries/counties/county_shapefiles_aea')
    _co_climate = os.path.join(gis, 'training_data', 'humid', 'county_precip_normals')
    _nass = os.path.join(gis, 'nass_data', 'nass_irr_crop_new.csv')
    _lc_meta = os.path.join(os.path.dirname(__file__), 'landcover.json')
    county_training_devlopment(state_fields=_state_fields, training_dir=_training_dir,
                               county_shape_wgs=_co_shapes_wgs,
                               county_fields_aea=_co_fields_wgs,
                               county_attr=_co_fields_attr,
                               county_shape_aea=_co_shapes_aea,
                               climate=_co_climate, landcover=_lc_meta)
# ========================= EOF ====================================================================
