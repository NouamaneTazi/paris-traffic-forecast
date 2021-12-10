from datetime import datetime
from pathlib import Path


def test_format_and_export_output(output_df, output_directory, name_of_the_group):
    output_columns = {"arc": object, "datetime": object, "debit_horaire": float, "taux_occupation": float}
    # 1. Check relevant columns are in output dataframe
    assert sorted(list(output_df.columns)) == list(
            output_columns.keys()
    ), "Some columns are missing or unnecessary columns are in output"
    # 2. Check types
    for col, col_type in output_columns.items():
        assert output_df[col].dtype == col_type, f"Column {col} does not have type {col_type}"
    # 3. Check datetime string has right format
    try:
        output_df.datetime.apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M"))
    except ValueError as e:
        raise e
    # 4. Check `arc` columns has right values
    assert sorted(list(output_df["arc"].unique())) == [
            "Champs-Elysées",
        "Convention",
        "Saint-Pères",
    ], "Output does not have expected unique values for column `arc`"
    # 5. Check dataframe has right number of rows
    assert output_df.shape[0] == 360, f"Expected number of rows is 360, output has {output_df.shape[0]}"
    # 6. Export output
    output_path = Path(output_directory) / f"output_{name_of_the_group}.csv"
    print(f"[SAVE OUTPUT] Saving output here: {output_path}")
    output_df[output_columns.keys()].to_csv(output_path, sep=";")


# test_format_and_export_output(output_df, "/Users/pansardcaroline/Desktop", "viguier_dekergaradec_pansard")
