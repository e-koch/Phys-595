
'''
Create text files of the selected SDSS samples (DR12)
'''

from astropy.io import fits
import numpy as np


table_hdu = fits.open("specObj-dr12.fits", ignore_missing_end=True)

z = table_hdu[1].data["Z"]
z_err = table_hdu[1].data["Z_ERR"]
ston = table_hdu[1].data["SN_MEDIAN_ALL"]
zwarn = table_hdu[1].data["ZWARNING"]
primspec = table_hdu[1].data["SPECPRIMARY"]


samples = np.logical_and(z < 0.36, z > 0.015)
samples = np.logical_and(z_err < 0.05, samples)
samples = np.logical_and(ston > 5, samples)
samples = np.logical_and(zwarn == 0, samples)
samples = np.logical_and(primspec == 1, samples)

good_samples = table_hdu[1].data[samples]

samples_hdu = \
    fits.HDUList([fits.PrimaryHDU(None, header=table_hdu[0].header),
                  fits.BinTableHDU(good_samples, header=table_hdu[1].header)])

samples_hdu.writeto("good_samples.fits")

del samples_hdu, samples, table_hdu

# Now create the download text files

from download_spectra import download_list

download_list("good_samples.fits", "all_samples_list.csv")
