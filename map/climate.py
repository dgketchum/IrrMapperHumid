import json
from pprint import pprint

import requests

requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

with open('/home/dgketchum/ncdc_noaa_token.json', 'r') as j:
    API_KEY = json.load(j)['auth']


def precip_anomaly(st_abv, co_fips, start_mo=4, end_month=9, start_year=1997, end_year=2021):
    url = 'https://www.ncdc.noaa.gov/cag/county/time-series/' \
          '{}-{}-pcp-{}-{}-{}-{}.json?' \
          'base_prd=true&begbaseyear=1901&endbaseyear=2000'.format(st_abv, co_fips, start_mo,
                                                                   end_month, start_year, end_year)
    response = requests.get(url=url)
    json_data = response.json() if response and response.status_code == 200 else None
    dct = {k[:4]: v['anomaly'] for k, v in json_data['data'].items()}
    return dct


if __name__ == '__main__':
    precip_anomaly('SD', 125)
# ========================= EOF ====================================================================
