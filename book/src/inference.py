def unmask(ds):
    actual_mask = ds['mask'].copy()
    ds = ds.fillna(1.0)
    ds['mask'].values[:] = 1.0
    ds = ds.fillna(1.0)
    return ds, actual_mask

def gen_inference_data_pipeline(ds, config):
    actual_shape = (
        len(ds['lat']),
        len(ds['lon']),
        config['data_config']['output_sequence_length']
    )
    
    config['data_config']['batch_dims'] = {
        'lat': len(ds['lat']),
        'lon': len(ds['lon'])
    }
    pipe = make_data_pipeline(
        ds=ds, 
        min_samples=0, 
        preload=True,
        filter_mask=False,
        **config['data_config']
    )
    return pipe, actual_shape


def run_model(model, ds, config):
    # Set up data
    ds, actual_mask = unmask(ds)
    pipe, actual_shape = gen_inference_data_pipeline(ds, config)
    
    # Run model
    predictions = []
    for i, (x, y) in tqdm(enumerate(pipe)):
        x = x.to(model.device)
        with torch.no_grad():
            yhat = model(x).cpu()
        yhat = yhat.reshape(actual_shape)
        predictions.append(yhat)
    # Put things back together
    # First get the start time and make a template
    start_time = config['data_config']['input_overlap']['time'] 
    template = test_data['pr'].isel(time=slice(start_time, None))
    # Now put the predictions into a DataArray
    swe_pred = xr.DataArray(
        torch.concat(predictions, dim=2).squeeze().cpu(),
        dims=('lat', 'lon', 'time'), 
    ) 
    # Use that to get the time span and assign a time coordinate
    selected_times = template.isel(time=slice(0, len(swe_pred['time'])))
    swe_pred = swe_pred.assign_coords(selected_times.coords)
    # Finally mask things out and return
    swe_pred = swe_pred.where(true_mask, drop=True)
    return swe_pred