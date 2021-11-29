import os
import sys
import json
from subprocess import check_call
from pprint import pprint

from mapco.state_county_names_codes import state_county_code, state_fips_code
from mapco.shape_ops import TEST_COUNTIES
from mapco.climate import get_climate

sys.path.append('/home/dgketchum/PycharmProjects/EEMapper')
from map.call_ee import request_band_extract, export_classification, is_authorized

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

os.environ['GDAL_DATA'] = 'miniconda3/envs/gcs/share/gdal/'


def county_classification(climate_dir):
    state_codes = state_fips_code()
    for s, c in state_codes.items():
        if s not in state_county_code().keys():
            continue
        for k, v in state_county_code()[s].items():
            geoid = v['GEOID']
            if geoid not in TEST_COUNTIES:
                continue
            dry_years = get_climate(geoid, climate_dir, return_dry=True, n_years=3)
            get_bands(geoid, s, dry_years)
            print(s, geoid, dry_years)


def get_bands(geoid, state, years, southern=False):
    try:
        pts = 'users/dgketchum/points/county/pts_{}'.format(geoid)
        geo = 'users/dgketchum/boundaries/{}'.format(state)
        file_ = 'bands_{}'.format(geoid)
        request_band_extract(file_prefix=file_, points_layer=pts, region=geo,
                             years=years, filter_bounds=True,
                             buffer=1e4, southern=southern, filter_years=False)
    except Exception as e:
        print(geoid, e)


def push_bands_to_asset(_dir, glob, state, bucket):
    shapes = []
    local_f = os.path.join(_dir, '{}_{}.csv'.format(state, glob))
    bucket = os.path.join(bucket, 'state_bands')
    _file = os.path.join(bucket, '{}_{}.csv'.format(state, glob))
    cmd = [GS, 'cp', local_f, _file]
    check_call(cmd)
    shapes.append(_file)
    asset_ids = [os.path.basename(shp).split('.')[0] for shp in shapes]
    ee_root = 'users/dgketchum/bands/state/'
    for s, id_ in zip(shapes, asset_ids):
        cmd = [EE, 'upload', 'table', '-f', '--asset_id={}{}'.format(ee_root, id_), s]
        check_call(cmd)
        print(id_, s)


def classify(out_coll, variable_dir, tables, years, glob, state, southern=False):
    vars = os.path.join(variable_dir, 'variables_{}_{}.json'.format(state, glob))
    with open(vars, 'r') as fp:
        d = json.load(fp)
    features = [f[0] for f in d[state]]
    var_txt = os.path.join(variable_dir, '{}_{}_vars.txt'.format(state, glob))
    with open(var_txt, 'w') as fp:
        for f in features:
            fp.write('{}\n'.format(f))
    table = os.path.join(tables, '{}_{}'.format(state, glob))
    geo = 'users/dgketchum/boundaries/{}'.format(state)
    export_classification(out_name=state, table=table, asset_root=out_coll, region=geo,
                          years=years, input_props=features, bag_fraction=0.5, southern=southern)
    pprint(features)


if __name__ == '__main__':
    is_authorized()
    gis = os.path.join('/media/research', 'IrrigationGIS')
    if not os.path.exists(gis):
        gis = '/home/dgketchum/data/IrrigationGIS'
    _bucket = 'gs://wudr'
    _co_climate = os.path.join(gis, 'training_data', 'humid', 'county_precip_normals')
    county_classification(_co_climate)
# ========================= EOF ====================================================================
