import os
import sys
from subprocess import check_call
from collections import OrderedDict

import numpy as np
import fiona
from shapely.geometry import shape
from pandas import read_csv

from state_county_names_codes import state_fips_code, state_county_code, county_acres
from cdl import cdl_crops
from climate import generate_precip_records, get_prec_anomaly

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

TEST_COUNTIES = ['31097', '46051', '20131', '38043', '40015', '29003', '20073', '46061', '20157', '46097',
                 '40031', '20061', '20011', '20051', '48147', '20165', '31127', '20125', '20205', '48459', '48257',
                 '20083', '40003', '38057', '38065', '20155', '48085', '20031', '20207', '20111', '20133',
                 '38067', '31053', '31027', '46013', '46125', '46005', '46103', '46039', '46087', '46073', '46099',
                 '29011', '38017', '38003', '20013', '38101', '20003', '20193',
                 '48397', '48429', '48363', '40115', '46095', '46119', '48113', '20163', '20053', '20045', '46101',
                 '40001', '20065', '20123', '40051', '40017', '40027', '46045', '20195', '20041', '40151', '20151',
                 '40047', '38069', '29013', '20141', '40105', '20105', '46111', '38103', '31067', '40011', '20025',
                 '46121', '48159', '40133', '40125', '20169', '40077', '46067', '20167', '20035', '38039',
                 '46079', '46027', '20147', '38031', '29145', '20159', '20085', '20161', '31051', '31133',
                 '48379', '20007', '38045', '20117', '20097', '40087', '48497', '38097', '48237',
                 '38037', '20029', '29037', '46105', '48203', '38073', '40069', '40075', '40093', '20049', '38025',
                 '38009', '40035', '38021', '31147', '31179', '20209', '48077', '29095', '20183', '31173',
                 '40143', '29119', '46017', '46037', '40043', '38029', '38075', '20107', '48277',
                 '20077', '20115', '48097', '29217', '31095', '20087', '38005', '40039', '46029', '46117', '40121',
                 '46031', '48067', '40013', '20139', '29005', '31023', '20127', '48223', '46075',
                 '48485', '20033', '38071', '38061', '38055', '20095', '46041', '48023', '46011',
                 '46071', '20197', '31153', '40053', '46021', '38015', '40145', '38085', '46023', '46055', '20001',
                 '20037', '48119', '40107', '40111', '31155', '20149', '20191', '40065', '40067',
                 '40141', '48417', '40083', '38051', '46035', '40049', '48467', '40109',
                 '40113', '20047', '38095', '31167', '29047', '29165', '48337', '40117', '40005', '40023', '40063',
                 '40091', '38091', '40061', '40079', '38081', '20005', '46091', '20145', '40101', '40135',
                 '38049', '48181', '31039', '38041', '20121', '20153', '40041', '46009', '46059', '48367',
                 '20015', '48449', '46065', '48387', '29097', '40089', '46085', '20039',
                 '20043', '20089', '20137', '40085', '40019', '40147', '48439', '31021', '48343', '20021', '31031',
                 '20057', '40123', '46069', '31037', '38089', '20009', '40029', '40033', '40071', '40127', '48487',
                 '40097', '20019', '31103', '20113', '38063', '40073', '40131', '46025', '31015', '31109', '31055',
                 '46083', '20079', '46109', '38035', '48009', '40103', '20177', '29087', '20091', '48503',
                 '20103', '38047', '20201', '46107', '31131', '48447', '46015', '40149', '38099', '29021',
                 '46043', '20179', '46115', '46127', '20017', '46123', '48423', '40021', '48231', '48121', '46077',
                 '20185', '38093', '31025', '46053', '31043', '46135', '20173', '40095', '46057', '48315', '20059',
                 '31177', '46049', '40099', '20135', '20099', '40153', '48037', '40081', '31159', '40037', '46089',
                 '20027', '38077', '48499', '46093', '38027', '20143', '40137', '48063', '46129', '38059',
                 '40119']


def run_counties(state_fields, training_dir, county_shape_wgs, county_fields_aea, county_attr,
                 county_shape_aea, climate, nass):
    state_codes = state_fips_code()
    for s, c in state_codes.items():
        if s not in state_county_code().keys():
            continue
        state_field_bounds = os.path.join(state_fields, '{}.shp'.format(s))
        for k, v in state_county_code()[s].items():
            geoid = v['GEOID']
            if geoid not in TEST_COUNTIES:
                continue
            # try:
            #     clip_field_boundaries_county(geoid, state_field_bounds, county_shape_wgs, county_fields_aea)
            #     attribute_county_fields(geoid, county_fields_aea, county_attr)
            get_climate(geoid, climate, return_dry=False)
            # write_potential_training_data(geoid, county_attr, training_dir, climate)
            # distribute_points(geoid, training_dir, n_points=4000, nass_data=nass, boundary_dir=county_shape_aea)
            # points_to_geographic(geoid, training_dir)
            # push_points_to_asset(geoid, training_dir, bucket=_bucket)
            # except Exception as e:
            #     print(s, geoid, e)


def points_to_geographic(geoid, _dir):
    target_dir = os.path.join(_dir, geoid)
    in_shp = os.path.join(target_dir, 'pts_{}_aea.shp'.format(geoid))
    out_shp = os.path.join(in_shp.replace('_aea', ''))
    cmd = [OGR, '-f', 'ESRI Shapefile', '-t_srs', WGS, '-s_srs', AEA, out_shp, in_shp]
    check_call(cmd)
    remove = [os.path.join(target_dir, x) for x in os.listdir(target_dir) if '_aea' in x and geoid in x]
    [os.remove(r) for r in remove]
    print(out_shp)


def get_climate(geoid, climate_dir, return_dry=True, n_years=3):
    precip_record = os.path.join(climate_dir, '{}.csv'.format(geoid))
    if not os.path.exists(precip_record):
        generate_precip_records(climate_dir, geoid)
        if not return_dry:
            return None
    precip_record = get_prec_anomaly(precip_record)
    dry_years = [x[0] for x in precip_record[:n_years]]
    if return_dry:
        return dry_years
    else:
        return precip_record


def popper(geometry):
    p = (4 * np.pi * geometry.area) / (geometry.boundary.length ** 2.)
    return p


def clip_field_boundaries_county(geoid, state_fields, county_shp_dir, out_dir):
    """clip field boundaries from state to county, project to albers equal area EPSG:102008"""
    co_field_filename = '{}.shp'.format(geoid)
    clip_src = os.path.join(county_shp_dir, co_field_filename)
    county_fields = os.path.join(out_dir, co_field_filename)
    cmd = [OGR, '-clipsrc', clip_src, '-s_srs', 'EPSG:4326', '-t_srs', AEA, county_fields, state_fields]
    check_call(cmd)
    print(geoid, 'clip')


def attribute_county_fields(geoid, county_shp_dir, out_co_dir, min_area=5e4):
    print(geoid, 'attribute')
    raw_fields = os.path.join(county_shp_dir, '{}.shp'.format(geoid))
    attr_fields = os.path.join(out_co_dir, '{}.shp'.format(geoid))
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


def write_potential_training_data(geoid, fields, training_dir, climate_dir, max_count=500):
    crops = cdl_crops().keys()
    c_dir = os.path.join(training_dir, geoid)
    if not os.path.isdir(c_dir):
        os.mkdir(c_dir)
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
        if prepend == 'p':
            meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
                [('FID', 'int:9'), ('YEAR', 'int:9')]), 'geometry': 'Polygon'}
        else:
            meta['schema'] = {'type': 'Feature', 'properties': OrderedDict(
                [('FID', 'int:9')]), 'geometry': 'Polygon'}
        with fiona.open(out_shp, 'w', **meta) as dst:
            for feat in features:
                if prepend == 'p':
                    f = {'type': 'Feature', 'properties': {'FID': ct, 'YEAR': dry_years[0]},
                         'geometry': feat['geometry']}
                else:
                    f = {'type': 'Feature', 'properties': {'FID': ct}, 'geometry': feat['geometry']}
                dst.write(f)
    print('{} dry years: {}, {} pivot, {} dry, {} uncult'.format(geoid, dry_years, p_ct, d_ct, u_ct))


def distribute_points(geoid, training_dir, n_points, nass_data, boundary_dir):
    df = read_csv(nass_data, index_col='GEOID')
    c_dir = os.path.join(training_dir, geoid)
    intersect_shape = os.path.join(boundary_dir, '{}.shp'.format(geoid))
    tot_acres = county_acres()[geoid]
    water = tot_acres['water']
    land = tot_acres['land']
    total = land + water
    irr_area = df.loc[int(geoid)]['IRR_2017']
    dry_area = df.loc[int(geoid)]['CROP_2017'] - irr_area
    irr_ratio = irr_area / total
    dry_ratio = dry_area / total
    uncult_ratio = (land - irr_area - dry_area) / total
    water_ratio = water / total

    wetlands = '/home/dgketchum/data/IrrigationGIS/compiled_training_data/aea/wetlands_22NOV2021.shp'
    if not os.path.exists(wetlands):
        wetlands = '/media/research/IrrigationGIS/compiled_training_data/aea/wetlands_22NOV2021.shp'
    kwargs = {
        'irrigated_path': os.path.join(c_dir, 'p_{}.shp'.format(geoid)),
        'uncultivated_path': os.path.join(c_dir, 'u_{}.shp'.format(geoid)),
        'unirrigated_path': os.path.join(c_dir, 'd_{}.shp'.format(geoid)),
        'wetland_path': wetlands,
    }

    kwargs.update({
        'irrigated': int(irr_ratio * n_points),
        'wetland': int(water_ratio * n_points),
        'uncultivated': int(uncult_ratio * n_points),
        'unirrigated': int(dry_ratio * n_points),
        'intersect': intersect_shape,
        'intersect_buffer': 100000})

    prs = dstpts.PointsRunspec(buffer=-20, **kwargs)
    out_name = os.path.join(c_dir, 'pts_{}_aea.shp'.format(geoid))
    prs.save_sample_points(out_name)


def push_points_to_asset(geoid, training_dir, bucket):
    c_dir = os.path.join(training_dir, geoid)
    local_files = [os.path.join(c_dir, 'pts_{}.{}'.format(geoid, ext)) for ext in
                   ['shp', 'prj', 'shx', 'dbf']]
    bucket = os.path.join(bucket, 'county_points')
    bucket_files = [os.path.join(bucket, 'pts_{}.{}'.format(geoid, ext)) for ext in
                    ['shp', 'prj', 'shx', 'dbf']]
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
    run_counties(state_fields=_state_fields, training_dir=_training_dir, county_shape_wgs=_co_shapes_wgs,
                 county_fields_aea=_co_fields_wgs, county_attr=_co_fields_attr, county_shape_aea=_co_shapes_aea,
                 climate=_co_climate, nass=_nass)
# ========================= EOF ====================================================================
