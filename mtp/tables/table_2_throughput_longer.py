import pandas as pd

from argparse import ArgumentParser


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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--raw-throughput-file", type=str, help="Path to jsonl file with raw throughput"
    )
    parser.add_argument(
        "--spec-throughput-file",
        type=str,
        help="Path to jsonl file with speculative throughput",
    )

    args = parser.parse_args()
    # Read JSONL file into a DataFrame
    df_raw = pd.read_json(args.raw_throughput_file, lines=True)
    stp_field = df_raw[df_raw["model"].str.contains("SingleTokenLM")].copy()
    stp_field.loc[0, "circuit"] = "STP"
    stp_field.loc[0, "ncomponent"] = 1
    stp_field["speedup"] = [1]

    df_spec = pd.read_json(args.spec_throughput_file, lines=True)

    df_spec = add_step_column(df_spec, "checkpoint")
    df_spec_best = df_spec[df_spec["step"] == 900]
    df_spec_best = df_spec_best[df_spec_best["ncomponent"].isin([32, 1])]
    df_spec_best = df_spec_best[df_spec_best["ntoken"].isin([1, 8, 16])]
    df_spec_best["circuit"] = df_spec_best["circuit"].str.replace("fully_factorized", "ff")

    agg_keys = {
            "avg_accepted_tokens": ["mean", "std", "count"],
            "avg_time_per_call": ["mean", "std", "count"],
            "tokens_per_second": ["mean", "std", "count"],
        }
    df_spec_best_agg = df_spec_best.groupby(["ntoken", "circuit", "ncomponent"]).agg(
            agg_keys
    )
    df_spec_best_agg = df_spec_best_agg.reset_index()

    df_collapsed = pd.DataFrame()
    df_collapsed["ntoken"] = df_spec_best_agg["ntoken"]
    df_collapsed["ncomponent"] = df_spec_best_agg["ncomponent"]
    df_collapsed["circuit"] = df_spec_best_agg["circuit"]
    df_collapsed["speedup"] = df_spec_best_agg[("tokens_per_second", "mean")] / stp_field["tokens_per_second"][0]


    for metric in agg_keys.keys():
        mean_col = (metric, "mean")
        std_col = (metric, "std")
        count_col = (metric, "count")

        # Auto-determine decimal places
        if metric == "avg_accepted_tokens":
            decimals = 2
        elif metric == "avg_time_per_call":
            decimals = 4
        else:
            decimals = 1


        # Create mean±std column
        df_collapsed[metric] = (
            df_spec_best_agg[mean_col].map(lambda x: f"{x:.{decimals}f}")
            + " \\scriptsize$\\pm$ "
            + df_spec_best_agg[std_col].map(lambda x: f"{x:.{decimals}f}")
        )

        # # Keep count
        # df_collapsed[f"{metric}_count"] = df_spec_best_agg[count_col]

    result = df_collapsed
    result = pd.concat([df_collapsed, stp_field[["circuit", "ntoken", "ncomponent", "avg_time_per_call", "tokens_per_second", "speedup"]]])
    result.sort_values(["ntoken", "ncomponent", "speedup"], inplace=True)
    result["circuit"] = result["circuit"].str.upper()
    colmap = {
        'ncomponent': '$r$',
        'ntoken': '$n$',
        'avg_accepted_tokens': '\\meanacc',
        'avg_time_per_call': '\\meanlat',
        'tokens_per_second': '\\meantoks',
        'tokens_per_second_no_spec': '\\maxtoks',
    }
    result = result.rename(columns=colmap)
    result = result[["$n$", "$r$", "circuit", "\\meanacc", "\\meanlat", "\\meantoks", "speedup"]]

    latex_table = result.to_latex(
        float_format="%4.2f",
        formatters={
            ("$r$"): lambda x: f"{x:<4d}",
            ("$n$"): lambda x: f"{x:<4d}",
            ("circuit"): lambda x: f"{x:<5s}",
        },
        index=False,
        label="tab:no-lora-stats"
    )
    print(latex_table)
