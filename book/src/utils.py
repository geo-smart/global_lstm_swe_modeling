import torch
import yaml
import warnings

def save_experiment(config, output_dir, name, model=None):
    outfile = f"{output_dir}/{name}.yml"
    if model:
        config["weights_file"] = f"{output_dir}/{name}.pt"
        torch.save(model.state_dict(), f"{output_dir}/{name}.pt")
    with open(outfile, "w") as f:
        yaml.dump(config, f)
    return outfile


def load_experiment(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f)
    # Convert tuples back to slices
    for period in ['test_period', 'valid_period', 'train_period']:
        try:
            time_range = config['data_config'][period]
            time_range = slice(*time_range)
            config['data_config'][period] = time_range
        except KeyError:
            warnings.warn(f"Missing key in config: {period}. Skipping...")
            continue
    return config