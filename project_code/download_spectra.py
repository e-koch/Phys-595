
'''
Download SDSS spectra from the plate ID and fibre
'''

import os
from astropy.io import fits

def download_spectra(plateid, fibreid, mjd, survey, download=False,
                     name_only=True):
    '''
    Download an SDSS spectrum given the parameters.
    '''

    # Convert to lowercase
    plateid = str(plateid)
    fibreid = str(fibreid)
    mjd = str(mjd)
    survey = survey.lower()

    # Pad with zeros
    plateid = '0' * (4 - len(plateid)) + plateid
    fibreid = '0' * (4 - len(fibreid)) + fibreid

    filename = "spec-"+plateid+"-"+mjd+"-"+fibreid+".fits"

    if survey == 'boss':
        file_suffix = "sdss/spectro/redux/v5_5_12/spectra/lite/"+plateid+"/"+filename
    elif survey == 'sdss' or survey == 'segue1':
        file_suffix = "sdss/spectro/redux/26/spectra/lite/"+plateid+"/"+filename
    elif survey == 'segue2':
        file_suffix = "sdss/spectro/redux/104/spectra/lite/"+plateid+"/"+filename
    else:
        raise NameError("Check the survey name. Inputted was %s" % (survey))

    if download:
        # Check to make sure that file exists
        os.system("wget http://data.sdss3.org/sas/dr10/"+file_suffix)

        try:
            fits.open(file_suffix)
        except IOError:
            pass

        return filename

    elif name_only:
        return filename

    else:
        return file_suffix


def download_list(obs_file, output_file):
    '''
    Makes a txt file with the locations of the files to download.
    '''

    # Open the file
    data_file = fits.open(obs_file)

    spectra_data = data_file[1].data
    del data_file

    num_spectra = spectra_data.size

    output = open(output_file, 'a')

    for i in range(num_spectra):
        spec_info = spectra_data[i]

        # Download the spectrum
        spec_name = \
            download_spectra(spec_info['PLATE'], spec_info['FIBERID'],
                             spec_info['MJD'], spec_info['SURVEY'],
                             download=False)

        output.write("%s\n" % spec_name)

    output.close()


def split_spectra_list(props_file, n_split=4):
    '''
    Take the list of spectra in the given file and split it into equal
    portions.
    '''

    from astropy.io import fits

    data = fits.open(props_file)

    header = data[0].header
    data = data[1].data

    num_spec = data['Z'].shape[0]

    num_split = num_spec / n_split
    left_split = [1] * (num_spec % n_split) + \
                 [0] * (n_split - num_spec % n_split)

    for num, rem in zip(range(1, n_split + 1), left_split):
        part = data[(num-1) * num_split:num * num_split + rem]

        filename = props_file[:-5] + "_part" + str(num) + ".fits"

        hdu = fits.HDUList([fits.PrimaryHDU(header), part])

        hdu.writeto(filename)

    return
