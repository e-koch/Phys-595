
'''
Create text files of the selected SDSS samples (DR12)
'''

from astropy.io import fits
import numpy as np

from download_spectra import download_list


table_hdu = fits.open("specObj-dr12.fits", ignore_missing_end=True)

classes = table_hdu[1].data["CLASS_NOSQO"]
z = table_hdu[1].data["Z_NOSQO"]
z_err = table_hdu[1].data["Z_ERR_NOSQO"]
ston = table_hdu[1].data["SN_MEDIAN_ALL"]
zwarn = table_hdu[1].data["ZWARNING_NOSQO"]
primspec = table_hdu[1].data["SPECPRIMARY"]

# Grab galaxies
samples = (classes == 'GALAXY')
print "Galaxy " + str(sum(samples)) + " out of " + str(classes.shape[0])
# Redshift limits
samples = np.logical_and(z < 0.36, samples)
print "z high " + str(sum(samples)) + " out of " + str(classes.shape[0])
samples = np.logical_and(z > 0.015, samples)
print "z low " + str(sum(samples)) + " out of " + str(classes.shape[0])
# Limit on redshift error, and ensure no 'bad' negative error
samples = np.logical_and(z_err < 0.05, samples)
print "z_err " + str(sum(samples)) + " out of " + str(classes.shape[0])
samples = np.logical_and(z_err > 0.0, samples)
print "z_err > 0 " + str(sum(samples)) + " out of " + str(classes.shape[0])
# S/N > 5 averaged over all frames
samples = np.logical_and(ston > 3, samples)
print "SN " + str(sum(samples)) + " out of " + str(classes.shape[0])
# No warning flags
samples = np.logical_and(zwarn == 0, samples)
print "WARN " + str(sum(samples)) + " out of " + str(classes.shape[0])
# Take best spectrum if there are duplicates of an object
samples = np.logical_and(primspec == 1, samples)

print("Found "+str(sum(samples))+" good samples out of "+str(z.shape[0]))

good_samples = table_hdu[1].data[samples]

samples_hdu = \
    fits.HDUList([fits.PrimaryHDU(None, header=table_hdu[0].header),
                  fits.BinTableHDU(good_samples, header=table_hdu[1].header)])

samples_hdu.writeto("good_samples_nosqo.fits")

del samples_hdu, samples

# Now create the download text files

download_list("good_samples_nosqo.fits", "all_samples_list_nosqo.csv",
              name_only=False)


classes = table_hdu[1].data["CLASS"]
z = table_hdu[1].data["Z"]
z_err = table_hdu[1].data["Z_ERR"]
ston = table_hdu[1].data["SN_MEDIAN_ALL"]
zwarn = table_hdu[1].data["ZWARNING"]
primspec = table_hdu[1].data["SPECPRIMARY"]

# Grab galaxies
samples = (classes == 'GALAXY')
print "Galaxy " + str(sum(samples)) + " out of " + str(classes.shape[0])
# Redshift limits
samples = np.logical_and(z < 0.36, samples)
print "z high " + str(sum(samples)) + " out of " + str(classes.shape[0])
samples = np.logical_and(z > 0.015, samples)
print "z low " + str(sum(samples)) + " out of " + str(classes.shape[0])
# Limit on redshift error, and ensure no 'bad' negative error
samples = np.logical_and(z_err < 0.05, samples)
print "z_err " + str(sum(samples)) + " out of " + str(classes.shape[0])
samples = np.logical_and(z_err > 0.0, samples)
print "z_err > 0 " + str(sum(samples)) + " out of " + str(classes.shape[0])
# S/N > 5 averaged over all frames
samples = np.logical_and(ston > 3, samples)
print "SN " + str(sum(samples)) + " out of " + str(classes.shape[0])
# No warning flags
samples = np.logical_and(zwarn == 0, samples)
print "WARN " + str(sum(samples)) + " out of " + str(classes.shape[0])
# Take best spectrum if there are duplicates of an object
samples = np.logical_and(primspec == 1, samples)

print("Found "+str(sum(samples))+" good samples out of "+str(z.shape[0]))

good_samples = table_hdu[1].data[samples]

samples_hdu = \
    fits.HDUList([fits.PrimaryHDU(None, header=table_hdu[0].header),
                  fits.BinTableHDU(good_samples, header=table_hdu[1].header)])

samples_hdu.writeto("good_samples.fits")

del samples_hdu, samples, table_hdu

# Now create the download text files

download_list("good_samples.fits", "all_samples_list.csv",
              name_only=False)
