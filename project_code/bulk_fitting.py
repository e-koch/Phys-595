
'''
Bulk spectral line fitting with SDSS galaxy spectra
'''

import os
from astropy.io import fits
from pandas import concat

# Bring in the package funcs
from specfit import do_specfit
from download_spectra import download_spectra


def bulk_fit(obs_file, output_file, keep_spectra=False):
    '''
    Downloads files based off of the entries in the given file, performs
    spectral line fitting and saves the results to a FITS table.
    '''

    # Open the file
    data_file = fits.open(obs_file)

    spectra_data = data_file[1].data
    del data_file

    num_spectra = spectra_data['Z'].shape[0]

    for i in range(num_spectra):
        spec_info = spectra_data[i]

        # Download the spectrum
        spec_name = \
            download_spectra(spec_info['PLATEID'], spec_info['FIBREID'],
                             spec_info['MJD'], spec_info['SURVEY'])

        spec_df = do_specfit(spec_name, verbose=False)

        if i == 0:
            df = spec_df
        else:
            df = concat([df, spec_df])

        if not keep_spectra:
            os.system('rm ' + spec_name)

    df.write(output_file)

    return
