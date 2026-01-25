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
    df_spec = pd.read_json(args.spec_throughput_file, lines=True)

    df_spec = add_step_column(df_spec, "checkpoint")
    df_spec_best = df_spec[df_spec["step"] == 900]
    df_spec_best = df_spec_best[df_spec_best["ntoken"] == 8]
    df_spec_best = df_spec_best[df_spec_best["circuit"].isin(["cp", "fully_factorized"])]
    df_spec_best["circuit"] = df_spec_best["circuit"].str.replace("fully_factorized", "ff")

    df_spec_best_agg = df_spec_best.groupby(["circuit", "ncomponent"]).agg(
        {
            "avg_accepted_tokens": ["mean", "std", "count"],
            "avg_time_per_call": ["mean", "std", "count"],
            "tokens_per_second": ["mean", "std", "count"],
        }
    )
    df_spec_best_agg = df_spec_best_agg.reset_index()

    df_raw = df_raw[df_raw["ntoken"] == 8]
    df_raw["circuit"] = df_raw["circuit"].str.replace("fully_factorized", "ff")
    df_raw = df_raw[df_raw["circuit"].isin(["cp", "ff"])]
    df_raw = df_raw[df_raw["adaptor"] == "none"]
    df_raw = df_raw[["ncomponent", "tokens_per_second"]]
    df_raw = df_raw.rename(columns={"tokens_per_second": "tokens_per_second_no_spec"})

    df_collapsed = pd.DataFrame()
    df_collapsed["circuit"] = df_spec_best_agg["circuit"].str.upper()
    df_collapsed["ncomponent"] = df_spec_best_agg["ncomponent"]
    df_collapsed.sort_values("ncomponent", inplace=True)

    for metric in ["avg_accepted_tokens", "avg_time_per_call", "tokens_per_second"]:
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

    result = pd.merge(df_collapsed, df_raw, on='ncomponent', how='inner')
    result = result.rename(columns={
        'ncomponent': '$r$',
        'avg_accepted_tokens': '\\meanacc~\\incfield',
        'avg_time_per_call': '\\meanlat~\\decfield',
        'tokens_per_second': '\\meantoks~\\incfield',
        'tokens_per_second_no_spec': '\\maxtoks',
    })

    latex_table = result.to_latex(
        float_format="%4.2f",
        formatters={
            ("$r$"): lambda x: f"{x:<4d}",
        },
        # multirow=False,
        index=False,
        label="tab:cp-vary-stats"
    )
    print(latex_table)
