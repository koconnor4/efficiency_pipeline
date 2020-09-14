"""
kfo 09/02/2020
using requests to get data from lco archive
https://developers.lco.global/
"""

import requests

API_TOKEN = '624ef1e356eeccc137f26dfe47bced1c1dd671fb'  # API token obtained from https://observe.lco.global/accounts/profile/

response = requests.get(
    'https://observe.lco.global/api/proposals/',
    headers={'Authorization': 'Token {}'.format(API_TOKEN)}
)

# Make sure this API call was successful
try:
    response.raise_for_status()
except requests.exceptions.HTTPError as exc:
    print('Request failed: {}'.format(response.content))
    raise exc

proposals_dict = response.json()  # API returns a json dictionary containing proposal information

print('Member of {} proposals'.format(proposals_dict['count']))

# Loop over each proposal and print some things about it.
for proposal in proposals_dict['results']:
    print('\nProposal: {}'.format(proposal['id']))
    for time_allocation in proposal['timeallocation_set']:
        print('{0:.3f} out of {1} standard hours used on instrument type {2} for semester {3}'.format(
            time_allocation['std_time_used'],
            time_allocation['std_allocation'],
            time_allocation['instrument_type'],
            time_allocation['semester'],
        ))

        
# lets get down to business to defeat the huns

# find the token I need to get data
requests.post(
    'https://archive-api.lco.global/api-token-auth/',
    data = {
        'username': 'oconnorf@email.sc.edu',
        'password': 'KfoPhy774'
    }
).json()
token = 'cb00e632ec494f78571af0b2f7db879e3546fb52'
# give a url with constraints on what you wanna dl (proposal,rlevel,object,start/end etc...)
# this example is grabbing all 08/31 banzai-reduced data for lco2020b (18 files)
url = 'https://archive-api.lco.global/frames/?q=a&RLEVEL=91&PROPID=LCO2020B-015&INSTRUME=&OBJECT=&SITEID=&TELID=&FILTER=&OBSTYPE=&EXPTIME=&BLKUID=&REQNUM=&basename=&start=2020-08-31%2000%3A00&end=2020-08-31%2023%3A59&id=&public=false'

tmp = requests.get(url,headers={'Authorization': 'Token {}'.format(token)})
response = tmp.json()
frames = response['results']
for frame in frames:
    with open(frame['filename'], 'wb') as f:
        f.write(requests.get(frame['url'],headers={'Authorization': 'Token {}'.format(token)}).content)

"""
# iterate over multiple pages? 
# not sure when/if this is relevant but the description implies maybe if trying to dl alot, which will eventually

frame_collection = []

def fetch_frames(url, collection):
    response = requests.get(url,headers={'Authorization': 'Token {}'.format(token)}).json()
    collection += response['results']
    if response.get('next'):
        fetch_frames(response['next'], collection)


fetch_frames(url, frame_collection)
print(len(frame_collection))
"""