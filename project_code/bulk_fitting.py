
'''
Bulk spectral line fitting with SDSS galaxy spectra
'''

import os
import shutil
from astropy.io import fits
from pandas import DataFrame, Series
import numpy as np

# Bring in the package funcs
from specfit import do_specfit
from download_spectra import download_spectra


def bulk_fit(obs_file, output_file, keep_spectra=True, split_save=True,
             num_save=10, num_start=-1):
    '''
    Downloads files based off of the entries in the given file, performs
    spectral line fitting and saves the results to a FITS table.

    Parameters
    ----------

    num_start : int {<num_save}
        Used for restarting at a point set by num_save. Default is -1, which
        starts at the beginning.
    '''

    # Open the file
    data_file = fits.open(obs_file)

    spectra_data = data_file[1].data
    del data_file

    num_spectra = spectra_data.size
    save_nums = [(num_spectra/num_save)*(i+1) for i in range(num_save)]
    save_nums[-1] = num_spectra-1
    save_nums.append(0)

    start_pt = save_nums[num_start]

    for i in range(start_pt, num_spectra):
        spec_info = spectra_data[i]

        # Download the spectrum
        spec_name = \
            download_spectra(spec_info['PLATE'], spec_info['FIBERID'],
                             spec_info['MJD'], spec_info['SURVEY'])

        spec_name = "spectra/" + spec_name

        try:
            spec_df = do_specfit(spec_name, verbose=False)
        except IOError:
            download_spectra(spec_info['PLATE'], spec_info['FIBERID'],
                             spec_info['MJD'], spec_info['SURVEY'],
                             download=True)
            try:
                shutil.move(spec_name.split("/")[-1], spec_name.split("/")[0])
                spec_df = do_specfit(spec_name, verbose=False)
            except IOError:
                print("Could not download requested file.")
                # Put in the default size (11 lines fitted).
                spec_df = Series([np.NaN] * 44)
        if i == start_pt:
            df = DataFrame(spec_df, columns=[spec_name[:-5]])
        else:
            df[spec_name[:-5]] = spec_df

        if split_save and i in save_nums:
            posn = [j for j, x in enumerate(save_nums) if x == i][0]
            df.to_csv(output_file[:-4]+"_"+str(posn+1)+".csv")
        if not keep_spectra:
            os.system('rm ' + spec_name)

    if not split_save:
        df.to_csv(output_file)

    return

if __name__ == "__main__":

    import sys

    input_file = str(sys.argv[1])
    output_file = str(sys.argv[2])

    restart_point = int(sys.argv[3])

    bulk_fit(input_file, output_file, num_start=restart_point)
