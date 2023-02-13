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
    mask = xr.open_dataset('https://esiptutorial.blob.core.windows.net/eraswe/mask_10k_household.zarr', engine='zarr')
    terrain = xr.open_dataset('https://esiptutorial.blob.core.windows.net/eraswe/processed_slope_aspect_elevation.zarr', engine='zarr')
    met_ds['mask'] = mask['sd'].rename({'latitude': 'lat', 'longitude': 'lon'})
    met_ds = xr.merge([met_ds, terrain])
    met_ds['mask'] = np.logical_and(~np.isnan(met_ds['elevation']), met_ds['mask']>0 ).astype(int)
    return met_ds

#@functional_datapipe("subset_regions")
class RegionalSubsetterPipe(IterDataPipe):
        
    def __init__(self, ds, selected_regions, repeat_region=10, preload=True):
        self.current_region = None
        self.ds = ds
        self.repeat_region = repeat_region
        self.selected_regions = [s for s in selected_regions 
                                 for _ in range(self.repeat_region)]
        self.preload = preload
        
    def select_region(self, region): 
        regions = regionmask.defined_regions.ar6.land
        region_id_mask = regions.mask(ds['lon'], ds['lat'])
        reg = np.unique(region_id_mask.values)
        reg = reg[~np.isnan(reg)]
        region_abbrevs = np.array(regions[reg].abbrevs)
        region_names = np.array(regions[reg].names)
        
        selection_mask = 0.0 * region_id_mask.copy()
        region_idx = np.argwhere(region_abbrevs == region)[0][0]
        region_mask = (region_id_mask == region_idx).astype(int)
        return self.ds.where(region_mask, drop=True)

    def __iter__(self):
        for region in self.selected_regions:
            if region != self.current_region:
                self.selected_ds = self.select_region(region)
                if self.preload:
                    self.selected_ds = self.selected_ds.load()
            self.current_region = region
            yield self.selected_ds


def filter_batch(batch):
    return batch.where(batch['mask']>0, drop=True)


def transform_batch(batch):
    scale_means = xr.Dataset()
    scale_means['mask'] = 0.0
    scale_means['swe'] = 0.0
    scale_means['pr'] = 0.00
    scale_means['tasmax'] = 295.0
    scale_means['tasmin'] = 280.0
    scale_means['elevation'] = 630.0
    scale_means['aspect_cosine'] = 0.0
    
    scale_stds = xr.Dataset()
    scale_stds['mask'] = 1.0
    scale_stds['swe'] = 3.0
    scale_stds['pr'] = 1/100.0
    scale_stds['tasmax'] = 80.0
    scale_stds['tasmin'] = 80.0
    scale_stds['elevation'] = 830.0
    scale_stds['aspect_cosine'] = 1.0
    
    batch = (batch - scale_means) / scale_stds
    return batch


def stack_split_convert(
    batch, 
    in_vars, 
    out_vars, 
    in_selectors={},
    out_selectors={},
    device=None,
    min_samples=200
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
        x = torch.tensor(x.values).float()
        y = torch.tensor(y.values).float()
        if device:
            x = x.to(device)
            y = y.to(device)
    else:
        x, y = torch.tensor([]), torch.tensor([])
    return x, y