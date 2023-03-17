import os
import torch
import intake
import regionmask
import xbatcher
import xarray as xr
import numpy as np
import zen3geo

from functools import partial
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterDataPipe

scale_means = xr.Dataset()
scale_means['mask'] = 0.0
scale_means['swe'] = 0.0
scale_means['cbrt_swe'] = 0.00
scale_means['pr'] = 0.00
scale_means['cbrt_pr'] = 0.00
scale_means['tasmax'] = 295.0
scale_means['tasmin'] = 280.0
scale_means['elevation'] = 630.0
scale_means['aspect_cosine'] = 0.0

scale_stds = xr.Dataset()
scale_stds['mask'] = 1.0
scale_stds['swe'] = 3.0
scale_stds['cbrt_swe'] = 1/2
scale_stds['pr'] = 1/100.0
scale_stds['cbrt_pr'] = 1/10.0
scale_stds['tasmax'] = 80.0
scale_stds['tasmin'] = 80.0
scale_stds['elevation'] = 830.0
scale_stds['aspect_cosine'] = 1.0


def get_static_data():
    base_url = 'https://esiptutorial.blob.core.windows.net/eraswe'
    mask = xr.open_dataset(
        f'{base_url}/mask_10k_household.zarr',  engine='zarr'
    )
    terrain = xr.open_dataset(
        f'{base_url}/processed_slope_aspect_elevation.zarr', engine='zarr'
    )
    terrain['mask'] = mask['sd'].rename({'latitude': 'lat', 'longitude': 'lon'})
    terrain['mask'] = np.logical_and(
        ~np.isnan(terrain['elevation']), 
        terrain['mask']>0 
    ).astype(int)   
    return terrain


def merge_data():
    era5_daily_cat = intake.open_esm_datastore(
        'https://cpdataeuwest.blob.core.windows.net/cp-cmip/training/ERA5-daily-azure.json'
    )
    met_files = sorted(list(era5_daily_cat.search(cf_variable_name='tasmax').df['zstore']))
    years = np.arange(1985, 2015)
    swe_files = [f'https://esiptutorial.blob.core.windows.net/eraswe/era5_raw_swe/era5_raw_swe_{year}.zarr'
             for year in years]
    swe_ds = xr.open_mfdataset(swe_files, engine='zarr')
    daily_swe = swe_ds.resample(time='1D').mean().rename({'latitude': 'lat', 'longitude': 'lon'})
    met_ds = xr.open_mfdataset(met_files,  engine='zarr')#.sel(time=swe_data['time'])
    met_ds = met_ds.sel(time=slice(daily_swe['time'].min(), daily_swe['time'].max()))
    met_ds['swe'] = daily_swe['sd']
    
    terrain = get_static_data()
    met_ds = xr.merge([met_ds, terrain])
    met_ds['mask'] = np.logical_and(~np.isnan(met_ds['elevation']), met_ds['mask']>0 ).astype(int)
    met_ds['cbrt_swe'] = np.power(met_ds['swe'], 1/3)
    met_ds['cbrt_pr'] = np.power(met_ds['pr'], 1/3)
    return met_ds


def select_region(ds, region): 
    # Get all regions & create mask from lat/lons
    regions = regionmask.defined_regions.ar6.land
    region_id_mask = regions.mask(ds['lon'], ds['lat'])
    # Get unique listing of region names & abbreviations
    reg = np.unique(region_id_mask.values)
    reg = reg[~np.isnan(reg)]
    region_abbrevs = np.array(regions[reg].abbrevs)
    region_names = np.array(regions[reg].names)
    # Create a mask that only contains the region of interest
    selection_mask = 0.0 * region_id_mask.copy()
    region_idx = np.argwhere(region_abbrevs == region)[0][0]
    region_mask = (region_id_mask == reg[region_idx]).astype(int)
    return ds.where(region_mask, drop=True)


class RegionalSubsetterPipe(IterDataPipe):
        
    def __init__(self, ds, selected_regions=None, preload=False):
        super().__init__()
        self.ds = ds
        self.selected_regions = self.to_sequence(selected_regions)
        self.preload = preload
        
    def to_sequence(self, seq):
        if isinstance(seq, str):
            return (seq, )
        return seq

    def __iter__(self):
        if not self.selected_regions:
            yield self.ds
        else:
            for region in self.selected_regions:
                self.selected_ds = select_region(self.ds, region)
                if self.preload:
                    self.selected_ds = self.selected_ds.load()
                yield self.selected_ds
            
            
def filter_batch(batch):
    return batch.where(batch['mask']>0, drop=True)


def transform_batch(
    batch, 
    scale_means=scale_means, 
    scale_stds=scale_stds
):
    batch = (batch - scale_means) / scale_stds
    return batch


def untransform_batch(
    batch, 
    scale_means=scale_means, 
    scale_stds=scale_stds
):
    batch = (batch * scale_stds) + scale_means
    return batch


def stack_split_convert(
    batch, 
    in_vars, 
    out_vars, 
    in_selectors={},
    out_selectors={},
    min_samples=50,
    dtype=torch.float32
):
    if len(batch['sample']) > min_samples:
        x = (batch[in_vars]
                 .to_array()
                 .transpose('sample', 'time', 'variable')
                 .isel(**in_selectors))
        y = (batch[out_vars]
                  .to_array()
                  .transpose('sample', 'time', 'variable')
                  .isel(**out_selectors))
        x = torch.tensor(x.values).to(dtype)
        y = torch.tensor(y.values).to(dtype)
    else:
        x, y = torch.tensor([], dtype=dtype), torch.tensor([], dtype=dtype)
    return x, y


def make_data_pipeline(
    ds, 
    regions, 
    input_vars, 
    output_vars,
    input_sequence_length,
    output_sequence_length,
    batch_dims,
    input_overlap,
    preload=False,
    min_samples=50,
    filter_mask=True,
    dtype=torch.float32,
    **kwargs
):
    # Preamble: just set some stuff up
    output_selector = {'time': slice(-output_sequence_length, None)}
    input_dims={'time': input_sequence_length}
    varlist = ['mask'] + input_vars + output_vars
    convert = partial(
        stack_split_convert, 
        in_vars=input_vars, 
        out_vars=output_vars, 
        out_selectors=output_selector,
        min_samples=min_samples,
        dtype=dtype,
    )
    # Chain together the datapipe
    dp = RegionalSubsetterPipe(
        ds[varlist], selected_regions=regions, preload=preload
    )
    dp = dp.slice_with_xbatcher(
        input_dims=input_dims,
        batch_dims=batch_dims,
        input_overlap=input_overlap,
        preload_batch=False
    )
    if filter_mask:
        dp = dp.map(filter_batch)
    dp = dp.map(transform_batch)
    dp = dp.map(convert)   
    return dp