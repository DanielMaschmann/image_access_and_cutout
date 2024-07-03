"""
simple script to access HST and JWST images, create cutouts and reproject them to a common grid
"""
import numpy as np
import matplotlib.pyplot as plt
import warnings

from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from reproject import reproject_interp

# constants need here:
# import speed of light in m per seconds
from scipy.constants import c as speed_of_light_mps
sr_per_square_deg = 0.00030461741978671  # steradians per square degree


# All tasks we want to perform here we write in functions

def load_img(file_name, hdu_number=0):
    """function to open hdu using astropy.

    Parameters
    ----------
    file_name : str or Path
        file name to open
    hdu_number : int or str
        hdu number which should be opened. can be also a string such as 'SCI' for JWST images

    Returns
    -------
    array-like,  ``astropy.io.fits.header.Header`` and ``astropy.wcs.WCS` and
    """
    # get hdu
    hdu = fits.open(file_name)
    # get header
    header = hdu[hdu_number].header
    # get WCS
    wcs = WCS(header)
    # update the header
    header.update(wcs.to_header())
    # reload the WCS and header
    header = hdu[hdu_number].header
    wcs = WCS(header)
    # load data
    data = hdu[hdu_number].data
    # close hdu again
    hdu.close()
    return data, header, wcs


def get_hst_img_conv_fct(img_header, img_wcs, flux_unit='Jy'):
    """
    get unit conversion factor to go from electron counts to mJy of HST images
    Parameters
    ----------
    img_header : ``astropy.io.fits.header.Header``
    img_wcs : ``astropy.wcs.WCS``
    flux_unit : str

    Returns
    -------
    conversion_factor : float

    """
    # convert the flux unit
    if 'PHOTFNU' in img_header:
        conversion_factor = img_header['PHOTFNU']
    elif 'PHOTFLAM' in img_header:
        # wavelength in angstrom
        pivot_wavelength = img_header['PHOTPLAM']
        # inverse sensitivity, ergs/cm2/Ang/electron
        sensitivity = img_header['PHOTFLAM']
        # speed of light in Angstrom/s
        c = speed_of_light_mps * 1e10
        # change the conversion facto to get erg s−1 cm−2 Hz−1
        f_nu = sensitivity * pivot_wavelength ** 2 / c
        # change to get Jy
        conversion_factor = f_nu * 1e23
    else:
        raise KeyError('there is no PHOTFNU or PHOTFLAM in the header')

    pixel_area_size_sr = img_wcs.proj_plane_pixel_area().value * sr_per_square_deg
    # rescale data image
    if flux_unit == 'Jy':
        # rescale to Jy
        conversion_factor = conversion_factor
    elif flux_unit == 'mJy':
        # rescale to mJy
        conversion_factor *= 1e3
    elif flux_unit == 'MJy/sr':
        # get the size of one pixel in sr with the factor 1e6 for the conversion of Jy to MJy later
        # change to MJy/sr
        conversion_factor /= (pixel_area_size_sr * 1e6)
    else:
        raise KeyError('flux_unit ', flux_unit, ' not understand!')

    return conversion_factor


def get_jwst_conv_fact(img_wcs, flux_unit='Jy'):
    """
    get unit conversion factor for JWST image observations
    ----------
    img_wcs : ``astropy.wcs.WCS``
    flux_unit : str

    Returns
    -------
    conversion_factor : float

    """
    pixel_area_size_sr = img_wcs.proj_plane_pixel_area().value * sr_per_square_deg
    # rescale data image
    if flux_unit == 'Jy':
        # rescale to Jy
        conversion_factor = pixel_area_size_sr * 1e6

    elif flux_unit == 'mJy':
        # rescale to Jy
        conversion_factor = pixel_area_size_sr * 1e9
    elif flux_unit == 'MJy/sr':
        conversion_factor = 1
    else:
        raise KeyError('flux_unit ', flux_unit, ' not understand')
    return conversion_factor


def get_img_cutout(img, wcs, coord, cutout_size):
    """function to cut out a region of a larger image with an WCS.
    Parameters
    ----------
    img : array-like
        (Ny, Nx) image
    wcs : ``astropy.wcs.WCS``
        astropy world coordinate system object describing the parameter image
    coord : ``astropy.coordinates.SkyCoord``
        astropy coordinate object to point to the selected area which to cutout
    cutout_size : float or tuple
        Units in arcsec. Cutout size of a box cutout. If float it will be used for both box length.
    Returns
    -------
    cutout :  ``astropy.nddata.Cutout2D`` object
        cutout object of the initial image
    """
    if isinstance(cutout_size, tuple):
        size = cutout_size * u.arcsec
    elif isinstance(cutout_size, float) | isinstance(cutout_size, int):
        size = (cutout_size, cutout_size) * u.arcsec
    else:
        raise KeyError('cutout_size must be float or tuple')

    # check if cutout is inside the image
    pix_pos = wcs.world_to_pixel(coord)
    if (pix_pos[0] > 0) & (pix_pos[0] < img.shape[1]) & (pix_pos[1] > 0) & (pix_pos[1] < img.shape[0]):
        return Cutout2D(data=img, position=coord, size=size, wcs=wcs)
    else:
        warnings.warn("The selected cutout is outside the original dataset. The data and WCS will be None",
                      DeprecationWarning)
        cut_out = type('', (), {})()
        cut_out.data = None
        cut_out.wcs = None
        return cut_out


def reproject_image(data, wcs, new_wcs, new_shape):
    """function to reproject an image with na existing WCS to a new WCS
    Parameters
    ----------
    data : array-like
    wcs : ``astropy.wcs.WCS``
    new_wcs : ``astropy.wcs.WCS``
    new_shape : tuple

    Returns
    -------
    new_data : array-like
        new data reprojected to the new wcs
    """
    hdu = fits.PrimaryHDU(data=data, header=wcs.to_header())
    return reproject_interp(hdu, new_wcs, shape_out=new_shape, return_footprint=False)


# Here you need to specify the data path on your system!
# data path to an arbitrary HST band:
file_path_arb_hst_band = ('/media/benutzer/Extreme Pro/data/phangs_hst/HST_reduced_images/ngc628/uvisf555w/'
                          'ngc628_uvis_f555w_exp_drc_sci.fits')
# data path to an arbitrary NIRCAM band:
file_path_arb_nircam_band = ('/media/benutzer/Extreme Pro/data/phangs_jwst/v1p1p1/ngc0628/'
                             'ngc0628_nircam_lv3_f335m_i2d_anchor.fits')

# open HST data
data_hst, header_hst, wcs_hst = load_img(file_name=file_path_arb_hst_band, hdu_number=0)
# now the HST data opened is in electron counts per second, and we do want to have a unified unit
# here we chose MJy per steradians
data_hst *= get_hst_img_conv_fct(img_header=header_hst, img_wcs=wcs_hst, flux_unit='MJy/sr')

# open Nircam data
data_nircam, header_nircam, wcs_nircam = load_img(file_name=file_path_arb_nircam_band, hdu_number='SCI')
# change to MJy per steradians
data_nircam *= get_jwst_conv_fact(img_wcs=wcs_nircam, flux_unit='MJy/sr')

# arbitrary cluster coordinates
ra_cluster = 24.160755002266622
dec_cluster = 15.765572739515399
# convert into sky-coordinate object
cluster_pos = SkyCoord(ra=ra_cluster*u.deg, dec=dec_cluster*u.deg)
# cutout size in arcseconds
size = (2, 2)

# get HST cutout
hst_cutout = get_img_cutout(img=data_hst, wcs=wcs_hst, coord=cluster_pos, cutout_size=size)
nircam_cutout = get_img_cutout(img=data_nircam, wcs=wcs_nircam, coord=cluster_pos, cutout_size=size)

# plot the cluster and look at it
fig, ax = plt.subplots(ncols=2)
ax[0].imshow(hst_cutout.data)
ax[1].imshow(nircam_cutout.data)
plt.show()

print('HST cutout shape: ', hst_cutout.data.shape)
print('NIRCAM cutout shape: ', nircam_cutout.data.shape)

# now reproject the nircam image to the same pixel grid as HST
# the output we get from this is no longer a cutout object and only a 2-D array.
# The corresponding WCS would be the HST WCS : hst_cutout.wcs
nircam_cutout_reprojected = reproject_image(data=nircam_cutout.data, wcs=nircam_cutout.wcs, new_wcs=hst_cutout.wcs,
                                            new_shape=hst_cutout.data.shape)

print('HST cutout shape: ', hst_cutout.data.shape)
print('Reprojected NIRCAM cutout shape: ', nircam_cutout_reprojected.shape)

# plot the cluster and look at it
fig_2, ax = plt.subplots(ncols=2)
ax[0].imshow(hst_cutout.data)
ax[1].imshow(nircam_cutout_reprojected)
plt.show()
