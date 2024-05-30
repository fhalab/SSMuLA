"""A script for saving dataframes as pngs"""

import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

from matplotlib.colors import LinearSegmentedColormap, to_rgb
import matplotlib.pyplot as plt
import seaborn as sns

# for html to png
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options

from SSMuLA.de_simulations import DE_TYPES
from SSMuLA.zs_analysis import ZS_OPTS, ZS_COMB_OPTS
from SSMuLA.vis_summary import ZS_METRICS
from SSMuLA.get_corr import LANDSCAPE_ATTRIBUTES, val_list, zs_list
from SSMuLA.vis import PRESENTATION_PALETTE_SATURATE, save_plt
from SSMuLA.util import checkNgen_folder

# Custom colormap for the MSE row, using greens
cmap_mse = LinearSegmentedColormap.from_list(
    "mse_cmap_r", ["#FFFFFF", "#9bbb59"][::-1], N=100
)  # dark to light green

# Create the colormap
custom_cmap = LinearSegmentedColormap.from_list(
    "bwg",
    [
        PRESENTATION_PALETTE_SATURATE["blue"],
        "white",
        PRESENTATION_PALETTE_SATURATE["green"],
    ],
    N=100,
)

geckodriver_path = "/disk2/fli/miniconda3/envs/SSMuLA/bin/geckodriver"

de_metrics = ["mean_all", "fraction_max"]

simple_des = {
    "recomb_SSM": "Recomb",
    "single_step_DE": "Single step",
    "top96_SSM": "Top96 recomb",
}

simple_de_metric_map = {}

for de_type in DE_TYPES:
    for de_metric in de_metrics:
        simple_de_metric_map[f"{de_type}_{de_metric}"] = simple_des[de_type]


# Styling the DataFrame
def style_dataframe(df):
    # Define a function to apply gradient selectively
    def apply_gradient(row):
        if row.name == "mse":
            # Generate colors for the MSE row based on its values
            norm = plt.Normalize(row.min(), row.max())
            rgba_colors = [cmap_mse(norm(value)) for value in row]
            return [
                f"background-color: rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]})"
                for rgba in rgba_colors
            ]
        else:
            return [""] * len(row)  # No style for other rows

    # Apply gradient across all rows
    styled_df = df.style.background_gradient(cmap="Blues")
    # Apply the custom gradient to the MSE row
    styled_df = styled_df.apply(apply_gradient, axis=1)
    return styled_df.format("{:.2f}").apply(
        lambda x: ["color: black" if x.name == "mse" else "" for _ in x], axis=1
    )


def styledf2png(
    df,
    filename,
    sub_dir="results/style_dfs",
    absolute_dir="/disk2/fli/SSMuLA/",
    width=800,
    height=1600,
):

    html_path = os.path.join(sub_dir, filename + ".html")
    checkNgen_folder(html_path)

    # Create a HTML file
    html_file = open(html_path, "w")
    html_file.write(df.to_html())
    html_file.close()

    options = Options()
    options.add_argument("--headless")  # Run Firefox in headless mode.

    s = Service(geckodriver_path)
    driver = webdriver.Firefox(service=s, options=options)

    driver.get(
        f"file://{os.path.join(absolute_dir, html_path)}"
    )  # Update the path to your HTML file

    # Set the size of the window to your content (optional)
    driver.set_window_size(width, height)  # You might need to adjust this

    # Take screenshot
    driver.save_screenshot(html_path.replace(".html", ".png"))
    driver.quit()


def get_lib_stat(
    lib_csv: str = "results/corr_all/384/boosting|ridge-top96/merge_all.csv",
):

    df = pd.read_csv(lib_csv)
    style_df = (
        df[["lib"] + LANDSCAPE_ATTRIBUTES]
        .set_index("lib")
        .T.style.format("{:.2f}")
        .background_gradient(cmap="Blues", axis=1)
    )

    return styledf2png(
        style_df,
        "lib_stat",
        sub_dir="results/style_dfs",
        absolute_dir="/disk2/fli/SSMuLA/",
        width=1450,
        height=950,
    )

def get_zs_zs_corr(
    corr_csv: str = "results/corr_all/384/boosting|ridge-top96/corr.csv",
    n_mut: str = "all",
    metric: str = "rho"
):
    df = pd.read_csv(corr_csv)

    simple_zs = [zs for zs in zs_list if n_mut in zs and metric in zs]

    style_df = (
        df[
            df["descriptor"].isin(
                simple_zs
            )
        ][["descriptor"] + simple_zs]
        .loc[df['descriptor'].isin(simple_zs)]
        .rename(columns={"descriptor": "ZS predictions"})
        .set_index("ZS predictions")
        .rename(index=lambda x: x.replace("double", "hd2"))
        .rename(columns=lambda x: x.replace("double", "hd2"))
        .style.format("{:.2f}")
        .background_gradient(cmap=custom_cmap, vmin=0, vmax=1)
    )

    return styledf2png(
        style_df,
        f"zs_{n_mut}_{metric}_heatmap_384-boosting|ridge-top96_zs",
        sub_dir="results/style_dfs",
        absolute_dir="/disk2/fli/SSMuLA/",
        width=1250,
        height=450,
    )



def get_zs_corr_ls(
    corr_csv: str = "results/corr_all/384/boosting|ridge-top96/corr.csv",
    n_mut: str = "all",
    metric: str = "rho"
):

    df = pd.read_csv(corr_csv)

    style_df = (
        df[
            df["descriptor"].isin(
                [zs for zs in zs_list if n_mut in zs and metric in zs]
            )
        ][["descriptor"] + LANDSCAPE_ATTRIBUTES]
        .rename(columns={"descriptor": "Landscape attributes", **simple_de_metric_map})
        .iloc[0:33]
        .set_index("Landscape attributes")
        .rename(index=lambda x: x.replace("double", "hd2")).T
        .style.format("{:.2f}")
        .background_gradient(cmap=custom_cmap, vmin=-1, vmax=1)
    )

    return styledf2png(
        style_df,
        f"zs_{n_mut}_{metric}_heatmap_384-boosting|ridge-top96_landscape_attributes",
        sub_dir="results/style_dfs",
        absolute_dir="/disk2/fli/SSMuLA/",
        width=1450,
        height=950,
    )


def get_zs_corr(
    corr_csv: str = "results/corr_all/384/boosting|ridge-top96/corr.csv",
    deorls: str = "de",
    de_calc: str = "mean_all",  # or fraction_max
    n_mut: str = "all",
):

    df = pd.read_csv(corr_csv)

    if deorls == "de":
        comp_list = [f"{de_type}_{de_calc}" for de_type in DE_TYPES]
        dets = de_calc
        width=625
    else:
        comp_list = LANDSCAPE_ATTRIBUTES
        dets = "landscape_attributes"
        width=1800

    style_df = (
        df[
            df["descriptor"].isin(
                [zs for zs in zs_list if n_mut in zs and "ndcg" not in zs]
            )
        ][["descriptor"] + comp_list]
        .rename(columns={"descriptor": "Landscape attributes", **simple_de_metric_map})
        .iloc[0:33]
        .set_index("Landscape attributes")
        .rename(index=lambda x: x.replace("double", "hd2"))
        .style.format("{:.2f}")
        .background_gradient(cmap=custom_cmap, vmin=-1, vmax=1)
    )

    return styledf2png(
        style_df,
        f"zs_{n_mut}_heatmap_384-boosting|ridge-top96_{dets}",
        sub_dir="results/style_dfs",
        absolute_dir="/disk2/fli/SSMuLA/",
        width=width,
        height=550,
    )


def get_corr_heatmap(
    corr_csv: str = "results/corr_all/384/boosting|ridge-top96/corr.csv",
    de_calc: str = "mean_all",  # or fraction_max
):

    de_list = [f"{de_type}_{de_calc}" for de_type in DE_TYPES]

    df = pd.read_csv(corr_csv)
    style_df = (
        df[["descriptor"] + de_list]
        .rename(columns={"descriptor": "Landscape attributes", **simple_de_metric_map})
        .iloc[0:33]
        .set_index("Landscape attributes")
        .style.format("{:.2f}")
        .background_gradient(cmap=custom_cmap, vmin=-1, vmax=1)
    )

    return styledf2png(
        style_df,
        f"corr_heatmap_384-boosting|ridge-top96_{de_calc}",
        sub_dir="results/style_dfs",
        absolute_dir="/disk2/fli/SSMuLA/",
        width=720,
        height=975,
    )


def get_importance_heatmap(
    lib_csv: str = "results/corr_all/384/boosting|ridge-top96/merge_all.csv",
    de_calc: str = "mean_all",  # or fraction_max
):

    de_list = [f"{de_type}_{de_calc}" for de_type in DE_TYPES]

    df = pd.read_csv(lib_csv)

    # Load your dataset
    # data = pd.read_csv('path_to_your_data.csv')

    # Select features and targets
    features = df[LANDSCAPE_ATTRIBUTES]
    targets = df[val_list]

    lr_df_list = []

    # Splitting the dataset for each target and fitting a model
    for target in targets.columns:
        lr_model = LinearRegression()
        lr_model.fit(features, df[target])

        # Feature importance
        feature_importances = pd.DataFrame(
            lr_model.coef_, index=LANDSCAPE_ATTRIBUTES, columns=[target]
        )

        lr_df_list.append(feature_importances)
    lr_df = pd.concat(lr_df_list, axis=1)
    lr_df.index.names = ["Landscape attributes"]

    style_df = (
        lr_df[de_list]
        .rename(columns=simple_de_metric_map)
        .style.format("{:.2f}")
        .background_gradient(cmap=custom_cmap)
    )

    return styledf2png(
        style_df,
        f"importance_heatmap_384-boosting|ridge-top96_{de_calc}",
        sub_dir="results/style_dfs",
        absolute_dir="/disk2/fli/SSMuLA/",
        width=720,
        height=975,
    )