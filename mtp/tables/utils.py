import os


def parse_filename(filename: str) -> dict[str, str]:
    """
    Extract attributes from a throughput benchmark filename.

    Args:
        filename: Name like 'throughput-sampling-evabyte-lora-continued-1024-250.jsonl'

    Returns:
        Dictionary with keys: mode, model, subset

    Example:
        >>> parse_filename('throughput-sampling-evabyte-lora-continued-1024-250.jsonl')
        {'mode': 'sampling', 'model': 'evabyte', 'subset': 'lora-continued'}
    """
    # Remove extension and prefix
    gpu = os.path.basename(os.path.dirname(filename))
    name = os.path.basename(filename)
    name = name.replace('.jsonl', '').replace('throughput-', '')

    # Split and extract
    parts = name.split('-')

    return {
        'mode': parts[0],
        'model': parts[1],
        'subset': f'{parts[2]}-{parts[3]}',
        'gpu': gpu.lower()
    }


def add_step_column(df, column_name, new_column_name="step"):
    """
    Extract the step value from entries containing 'model@{number}.pt'

    Parameters:
    df: pandas DataFrame
    column_name: name of the column containing the model strings
    new_column_name: name for the new column (default: 'step')

    Returns:
    pandas DataFrame with new column containing step values
    """
    # Use regex to find 'model@' followed by digits
    pattern = r"@(\d+)"
    df[new_column_name] = (
        df[column_name].str.extract(pattern, expand=False).astype("Int64")
    )

    return df
