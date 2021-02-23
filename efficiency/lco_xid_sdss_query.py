from astropy.table import Table
from astropy.io import ascii
import pandas as pd
import numpy as np

from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord

visibility = ascii.read('Visibility.csv')

def lco_xid_sdss_query():
    # match ra/dec of lco targets to recovered stars from sdss 
    # 112 sources
    Source_IDs = visibility['col1'] # Source IDs
    ra_deg = visibility['col2']
    dec_deg = visibility['col3'] 
    for idx in range(1,len(visibility[1:])+1):
        print("------------------------------")
        obj,ra,dec = Source_IDs[idx],ra_deg[idx],dec_deg[idx]
        print(idx,obj,ra,dec)
        
        """
        full_radius ~ pixscale * 2048 is arcsec from center of an LCO exposure image
        go to 90% of that radius to account for target ra/dec dithers i.e. not being perfectly centered and edge effects
        """
        full_radius = 0.389*(4096/2)    
        radius = 0.85*full_radius
        strradius = str(radius) + ' arcsec'
        print(radius,'ra ~ [{:.2f},{:.2f}], dec ~ [{:.2f},{:.2f}]'.format(float(ra)-radius/3600,float(ra)+radius/3600,float(dec)-radius/3600,float(dec)+radius/3600))
        fields = ['ra','dec','objid','run','rerun','camcol','field','r','mode','nChild','type','clean','probPSF',
                 'psfMag_r','psfMagErr_r'] 
        pos = SkyCoord(ra,dec,unit="deg",frame='icrs')
        xid = SDSS.query_region(pos,radius=strradius,fields='PhotoObj',photoobj_fields=fields) 
        Star = xid[xid['probPSF'] == 1]
        Gal = xid[xid['probPSF'] == 0]
        print(len(xid),len(Star),len(Gal))
        Star = Star[Star['clean']==1]
        print(len(Star))
        ascii.write(Star,f"{obj}_SDSS_CleanStar.csv")
        
        idx+=1
if __name__ == "__main__":
    lco_xid_sdss_query()