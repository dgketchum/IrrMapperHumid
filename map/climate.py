import os
import json
from datetime import date
from calendar import monthrange

from pandas import DataFrame, to_datetime, read_csv, date_range, concat
import requests

# requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

with open('/home/dgketchum/ncdc_noaa_token.json', 'r') as j:
    API_KEY = json.load(j)['auth']


def generate_precip_records(climate_dir, co_fip='46125'):
    header = {"token": API_KEY}
    params = {'locationid': 'FIPS:{}'.format(co_fip),
              'limit': 52,
              'datasetid': 'GHCND'}

    server = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/'
    url = server + 'stations'

    resp = requests.get(url=url, headers=header, params=params)
    jsn = resp.json()
    min_dates = [int(x['mindate'][:4]) for x in jsn['results']]
    idx = min_dates.index(min(min_dates))
    target = jsn['results'][idx]['id']
    dates, values = [], []
    for s in [1997, 2007, 2017]:
        params = {'stationid': target,
                  'limit': 1000,
                  'units': 'metric',
                  'datasetid': 'GSOM',
                  'datatypeid': 'PRCP',
                  'startdate': '{}-01-01'.format(s),
                  'enddate': '{}-12-31'.format(s + 9)}
        url = server + 'data'
        resp = requests.get(url=url, headers=header, params=params)
        res = resp.json()['results']
        dates = dates + [x['date'] for x in res]
        values = values + [x['value'] for x in res]

    df = DataFrame(data={'prec': values})
    df.index = to_datetime(dates)
    params = {'stationid': target,
              'limit': 1000,
              'startdate': '2010-01-01',
              'enddate': '2010-12-01',
              'datatypeid': 'MLY-PRCP-NORMAL',
              'units': 'metric',
              'datasetid': 'NORMAL_MLY'}

    url = server + 'data'
    resp = requests.get(url=url, headers=header, params=params)
    res = resp.json()['results']
    dt_range = date_range('1901-01-1', '1901-12-31', freq='M')
    values = [x['value'] for x in res]
    ndf = DataFrame(data={'prec': values}, index=dt_range)
    df = concat([df, ndf], axis=0, ignore_index=False)
    out_csv = os.path.join(climate_dir, '{}.csv'.format(co_fip))
    df.to_csv(out_csv)
    return None


def get_prec_anomaly(csv, start_month=4, end_month=9):
    df = read_csv(csv)
    s, e = df.index[0].year, df.index[-1].year
    dates = [(date(y, start_month, 1), date(y, end_month, monthrange(y, end_month)[1])) for y in range(s, e + 1)]
    prec = [df['prec'][d[0]: d[1]].sum() for d in dates]


if __name__ == '__main__':
    generate_precip_records()
# ========================= EOF ====================================================================
