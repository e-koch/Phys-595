
'''
Create text files of the selected SDSS samples (DR12)
'''

from astropy.io import fits
import numpy as np


table_hdu = fits.open("specObj-dr12.fits", ignore_missing_end=True)

classes = table_hdu[1].data["CLASS_NOQSO"]
z = table_hdu[1].data["Z_NOQSO"]
z_err = table_hdu[1].data["Z_ERR_NOQSO"]
ston = table_hdu[1].data["SN_MEDIAN_ALL"]
zwarn = table_hdu[1].data["ZWARNING_NOQSO"]
primspec = table_hdu[1].data["SPECPRIMARY"]

# Grab galaxies
samples = (classes == 'GALAXY')
# Redshift limits
samples = np.logical_and(z < 0.36, samples)
samples = np.logical_and(z > 0.015, samples)
# Limit on redshift error, and ensure no 'bad' negative error
samples = np.logical_and(z_err < 0.05, samples)
samples = np.logical_and(z_err > 0.0, samples)
# S/N > 5 averaged over all frames
samples = np.logical_and(ston > 5, samples)
# No warning flags
samples = np.logical_and(zwarn == 0, samples)
# Take best spectrum if there are duplicates of an object
samples = np.logical_and(primspec == 1, samples)

good_samples = table_hdu[1].data[samples]

samples_hdu = \
    fits.HDUList([fits.PrimaryHDU(None, header=table_hdu[0].header),
                  fits.BinTableHDU(good_samples, header=table_hdu[1].header)])

samples_hdu.writeto("good_samples.fits")

del samples_hdu, samples, table_hdu

# Now create the download text files

from download_spectra import download_list

download_list("good_samples.fits", "all_samples_list.csv",
              name_only=False)
