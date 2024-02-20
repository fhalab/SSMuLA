import pandas as pd
import numpy as np
import pathlib
from tqdm import tqdm
import ninetysix as ns
import holoviews as hv
import colorcet as cc
hv.extension('bokeh')

import panel as pn
pn.extension('tabulator')
import hvplot.pandas

from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

import os

def tidy_kinetics(
    file,
    sheet=0,
    controls=None,
    signal_label=None,
    ref_label=None,
):
    """Reads in data from a Tecan Spark Kinetic Stacker run.
    
    Parameters:
    -----------
    """
    _df = pd.read_excel(file, sheet_name=sheet, header=None)
    
    end_idx = _df[_df.iloc[:,0] == 'End Time'].index[-1]
    
    # TODO: Get labels from metadata
    label_idx = _df[_df.iloc[:,0] == 'List of actions in this measurement script:'].index[0]
    end_label_idx = _df[_df.iloc[:,0] == 'Name'].index[0] - 1
    _labels = _df.iloc[label_idx:end_label_idx].apply(lambda row: [col for col in row if not pd.isna(col)], axis=1)
    labels = [val[1] for val in _labels if 'Absorbance' in val]
    
    if signal_label is not None:
        if signal_label not in labels:
            raise ValueError(
                f"'{signal_label}' not found in instrument labels: {labels}."
            )
        
        signal_idx = int(_df[_df.iloc[:,0] == signal_label].index[0])
        
        # Check if it's the last label; if not, use labels[i+1] for end_idx
    
        if signal_label != labels[-1]:
            end_label = labels[labels.index(signal_label)+1]
            end_idx = _df[_df.iloc[:,0] == end_label].index[0]
        
        nrows = (end_idx-1) - (signal_idx+2)
        # print(signal_idx)
        # Get the subset dataframe
        df_sig = pd.read_excel(file, sheet_name=sheet, skiprows=signal_idx+1, nrows=nrows-2, header=[0,1,2]).T.reset_index()
        df_sig.columns = df_sig.iloc[0]
        
        # TODO: Make sure that only one label was obtained here
        df_sig = df_sig.drop(0, axis=0).reset_index(drop=True)
        
        # Rename
        rename_dict = {
            'Cycle Nr.': 'Cycle',
            'Time [s]': 'Time (s)',
            'Temp. [°C]': 'Temperature (°C)'
        }
        df_sig = df_sig.rename(columns=rename_dict)
        
        df_sig = df_sig.melt(id_vars=['Cycle', 'Time (s)', 'Temperature (°C)'], var_name='Well', value_name='Absorbance')
        
        df_sig['Row'] = [well[0] for well in df_sig['Well']]
        df_sig['Column'] = [int(well[1:]) for well in df_sig['Well']]
        
        if controls is not None:
            df_sig['Type'] = df_sig['Well'].map(controls)
            df_sig['Type'] = df_sig['Type'].replace({np.nan: 'Variant'})
        
        # TODO: Make this a checker function in util
        # Make sure columns are appropriate dtypes
        col_types = (
            ('Cycle', int),
            ('Time (s)', float),
            ('Temperature (°C)', float),
            ('Well', str),
            ('Absorbance', float)
        )
        
        for col, dtype in col_types:
            try:
                df_sig[col] = df_sig[col].astype(dtype)
            except KeyError:
                pass
                
        
        return df_sig

def bootstrap_samples(samples):
    centerpoint = np.mean(samples)
    new_samples = np.random.choice(samples, (10000,len(samples)))
    new_means = new_samples.mean(axis=1)
    sorted_means = np.sort(new_means)
    
    return (round(centerpoint,5), round(sorted_means[240],5), round(sorted_means[9750],5))

def setup_data(folder, file, well_info):

    # set up the data (basic)
    ecoli_data = tidy_kinetics(os.path.join(folder,file), signal_label='OD600')
    ecoli_data['Time (min)'] = ecoli_data['Time (s)'].apply(lambda x: x/60)
    plate = ns.Plate(ecoli_data, value_name='Absorbance', annotate={'controls':well_info}, zero_padding=True)
    
    # add a background-subtracted column
    temp_df = plate.df.groupby(['controls','Cycle']).aggregate(np.mean)
    temp_df = temp_df.reset_index()
    temp_df = temp_df.loc[temp_df['controls'] == 'negative']
    cycles = temp_df['Cycle'].values
    bg_absorb = temp_df['Absorbance'].values
    temp_dict = dict(zip(cycles, bg_absorb))

    temp_plate = plate.df.copy()
    temp_plate['bg_subtracted_Absorbance'] = temp_plate.apply(lambda row: row['Absorbance'] - temp_dict[row['Cycle']], axis=1)
    plate = ns.Plate(temp_plate)

    return plate

    # Get codon summary statistics
def summary_stats(codon_df, AA_df, parent_dict, library):

    # library
    print(f'Library {library}')
    print('----------\n')

    # codon calculations
    print('Codon stats:')
    print('------------')
    print('for 32 unique codons, expect', 32*32*32)
    print('for 64 unique codons, expect', 64*64*64)

    total_codons = len(codon_df['count'])
    n_parent = codon_df.loc[(codon_df['codon1'] == parent_dict['codons']['codon1']) & (codon_df['codon2'] == parent_dict['codons']['codon2']) & (codon_df['codon3'] == parent_dict['codons']['codon3'])]['count'].values[0]
    total_reads = codon_df['count'].values.sum()
    percent_parent = round(n_parent/total_reads*100, 3)

    print(f'{total_codons} total unique codons sets')
    print(f'{n_parent} sets of parent codons')
    print(f'{total_reads} total reads')
    print(f'{percent_parent}% parent codons (parent/total)')
    print(f'Parent is ~{n_parent/(total_reads/total_codons)}x oversampled')
    
    # Amino acid calculations
    print('\nAmino acid stats:')
    print('-----------------')
    print('for 21 unique AAs, expect', 21*21*21)

    total_AAs = len(AA_df['count'])
    n_parent_AAs = AA_df.loc[(AA_df['AA1'] == parent_dict['AAs']['codon1']) & (AA_df['AA2'] == parent_dict['AAs']['codon2']) & (AA_df['AA3'] == parent_dict['AAs']['codon3'])]['count'].values[0]
    total_AA_reads = AA_df['count'].values.sum()
    percent_parent_AAs = round(n_parent_AAs/total_AA_reads*100, 3)
    max_AAs = AA_df['count'].values.max()

    print(f'{total_AAs} total unique AA sets')
    print(f'{n_parent_AAs} sets of parent AAs')
    print(f'{max_AAs} sets of most common AAs')
    print(f'{total_AA_reads} total reads') # should match codon number!
    print(f'{percent_parent_AAs}% parent AAs')
    print(f'Parent is ~{n_parent_AAs/(total_AA_reads/total_AAs)}x oversampled')

def calculate_OD(
    row,
    OD_dict,
    replicate
):
    variant_OD = row[f'OutputFreq_{replicate}']*OD_dict[f'rep_{replicate}'][row['Time (h)']]

    return variant_OD

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return y

def plot_and_fit_sigmoid(x_data, y_data, k_input):
    
    p0 = [max(y_data), (np.max(x_data)+np.min(x_data))/2, k_input, min(y_data)]
    popt, pcov = curve_fit(sigmoid, x_data, y_data, p0, method='dogbox')

    L, x0, k, b = popt
    
    # plot
    x = np.arange(1,36,1)
    y = sigmoid(x, L, x0, k, b)

    return (hv.Curve((x,y)).opts(color='orange'), popt)

def estimate_growthrate(
    data
):
    return None

def populate_df_dict(
    df_files: dict, 
    df_folder: str, 
    seq_type='AAs'
    ) -> dict:
    """_summary_

    Args:
        df_files (dict): _description_
        df_folder (str): _description_
        seq_type (str, optional): _description_. Defaults to 'AAs'.

    Returns:
        dict: _description_
    """
    # Get the file paths with pathlib
    file_paths = pathlib.Path(df_folder).glob(f'*{seq_type}.csv')

    # loop through all of the globbed paths
    for file_path in tqdm(file_paths):

        file = file_path.stem
        library, file_date, _, _ = tuple(file.split('_'))
        df_files[library.split('Lib')[-1]] = pd.read_csv(file_path, index_col=0)
    
    return df_files

def loglinear(t, k, OD_init):
    """This function takes a t value or vector, a k value, and an initial OD value and calculates a log(OD) for each provided t value. The function is of the form log(OD) = kt + log(OD_init).

    Args:
        t (int, float, array): Single time or array of time values for which to
            calculate a single value or an array of values for log(OD).
        k (float): slope of the line mapping t to log(OD)
        OD_init (float): log(OD_init) is the y-intercept of the line mapping t
            to log(OD)

    Returns:
        float or array: single value or set of log(OD) values that map to the
            provided t values
    """
    log_OD = np.log(OD_init) + k*t
    
    return log_OD

def fit_loglinear(x_data: np.array, y_data: np.array, k_input: float):
    """_summary_

    Args:
        x_data (np.array): _description_
        y_data (np.array): _description_
        k_input (float): _description_
    """
    log_y_data = np.log(y_data)
    p0 = [k_input, np.min(y_data)]
    popt, pcov = curve_fit(loglinear, x_data, log_y_data, p0)

    return(popt)

############# GENERAL FUNCTIONS #############

def count_sequence(row, sequences, seq_type):
    """
    Returns the number of stop codons in a given row
    """
    seq_count = 0
    
    if seq_type == 'codon':
        seq_list = row[f'{seq_type}s'].split('_')

    elif seq_type == 'AA':
        seq_list = list(row[f'{seq_type}s'])

    else:
        raise ValueError('seq_type must be codon or AA')

    for x in seq_list:
        if x in sequences:
            seq_count += 1
            
    return seq_count

def compute_z_score(value, dist_mean, dist_std):
    """
    Computes the z-score of a given value given a distribution mean and std
    """
    return (value - dist_mean) / dist_std

def describe_data(
    df, 
    seq_type, 
    OD_dict, 
    min_input_count_val=0, 
    timepoints='all'  
):
    
    df = df[df['avg_InputCount'] >= min_input_count_val].copy()

    df['mu_1'] = df.apply(
        calc_mu,
        outputfreq_col = 'OutputFreq_1',
        inputfreq_col = 'avg_InputFreq',
        OD_dict = OD_dict,
        axis=1
    )

    df['mu_2'] = df.apply(
        calc_mu,
        outputfreq_col = 'OutputFreq_2',
        inputfreq_col = 'avg_InputFreq',
        OD_dict = OD_dict,
        axis=1
    )
    
    if seq_type == 'AA':
        df['# Stop'] = df.apply(
            count_sequence,
            sequences = ['*'],
            seq_type = 'AA',
            axis = 1
        )
    elif seq_type == 'codon':
        df['# Stop'] = df.apply(
            count_sequence,
            sequences = ['TAG'],
            seq_type = 'codon',
            axis = 1
        )

    # only use timepoints of interest!
    if timepoints == 'all':
        pass
    else:
        df = df[df['Time (h)'].isin(timepoints)].copy()

    # df['avg_mu'] = df[['mu_1', 'mu_2']].mean(axis=1)

    # df['z_score(mu)'] = df['avg_mu'].apply(
    #     compute_z_score,
    #     dist_mean = df[df['# Stop'] > 0]['avg_mu'].mean(),
    #     dist_std = df[df['# Stop'] > 0]['avg_mu'].std()
    # )

    # print(f'{len(df[df["z_score(mu)"] > 1.96])/len(df)*100}% of sequences are significantly better than background')

    # df['mu_1-bg'] = df['mu_1'] - df[df['# Stop']>0]['mu_1'].mean()
    # df['mu_2-bg'] = df['mu_2'] - df[df['# Stop']>0]['mu_2'].mean()

    # df['avg_mu-bg'] = df[['mu_1-bg', 'mu_2-bg']].mean(axis=1)

    return df

def simplify_data_and_describe(
    df, 
    seq_type, 
    OD_dict, 
    filter=True, 
    timepoints='all'
):
    
    if filter:
        df = df[df['avg_InputCount'] >= 10].copy()

    df['mu_1'] = df.apply(
        calc_mu,
        outputfreq_col = 'OutputFreq_1',
        inputfreq_col = 'avg_InputFreq',
        OD_dict = OD_dict,
        axis=1
    )

    df['mu_2'] = df.apply(
        calc_mu,
        outputfreq_col = 'OutputFreq_2',
        inputfreq_col = 'avg_InputFreq',
        OD_dict = OD_dict,
        axis=1
    )
    
    if seq_type == 'AA':
        df['# Stop'] = df.apply(
            count_sequence,
            sequences = ['*'],
            seq_type = 'AA',
            axis = 1
        )
    elif seq_type == 'codon':
        df['# Stop'] = df.apply(
            count_sequence,
            sequences = ['TAG'],
            seq_type = 'codon',
            axis = 1
        )

    # only use timepoints of interest!
    if timepoints == 'all':
        pass
    else:
        df = df[df['Time (h)'].isin(timepoints)].copy()

    df = df.groupby([f'{seq_type}{x}' for x in ['s', 1, 2,3,4]])['InputCount_1', 'InputCount_2', 'avg_InputCount', 'avg_InputFreq', 'mu_1', 'mu_2', '# Stop'].mean().reset_index().copy()

    df['avg_mu'] = df[['mu_1', 'mu_2']].mean(axis=1)

    df['z_score(mu)'] = df['avg_mu'].apply(
        compute_z_score,
        dist_mean = df[df['# Stop'] > 0]['avg_mu'].mean(),
        dist_std = df[df['# Stop'] > 0]['avg_mu'].std()
    )

    print(f'{len(df[df["z_score(mu)"] > 1.96])/len(df)*100}% of sequences are significantly better than background')

    df['mu_1-bg'] = df['mu_1'] - df[df['# Stop']>0]['mu_1'].mean()
    df['mu_2-bg'] = df['mu_2'] - df[df['# Stop']>0]['mu_2'].mean()

    df['avg_mu-bg'] = df[['mu_1-bg', 'mu_2-bg']].mean(axis=1)

    return df

#################### FITNESS CALCULATIONS ####################

def calc_enrich(row, outputfreq_col: str, inputfreq_col: str):
    """
    Enrichment = freq_fi / freq_0i
    """
    
    # freq_0i
    freq_init = row[inputfreq_col]
    
    # freq_fi
    freq_final = row[outputfreq_col]
    
    # Enrichment
    enrich = freq_final / freq_init
    
    return enrich

def calc_mu(row, outputfreq_col: str, inputfreq_col: str, OD_dict: dict, t_i:int=0):
    """
    mu_i = ln(x_fi / x_0i) * 1/(t_f-t_i)
    """

    # general info
    rep = outputfreq_col.split('_')[-1]

    # t (time)
    t_f = row['Time (h)']

    # x_0i
    OD_init = OD_dict[f'rep_{rep}'][t_i]
    freq_init = row[inputfreq_col]
    x_0i = OD_init * freq_init

    # x_fi
    OD_final = OD_dict[f'rep_{rep}'][t_f]
    freq_final = row[outputfreq_col]
    x_fi = OD_final * freq_final
    
    # mu_i
    mu_i = np.log(x_fi / x_0i) * 1/(t_f-t_i)

    return mu_i

def get_background_dict(data: pd.DataFrame, bg_AAs: str, n_AAs: int):
    """
    Get background data for each position
    """

    assert n_AAs == len(bg_AAs), 'n_AAs does not match length of bg_AAs'

    background_df = data[data['AA_Combo'] == bg_AAs].copy().reset_index(drop=True)

    background_dict = background_df.set_index(['Time (h)']).T.to_dict()

    return background_dict

def calc_mu_ratio(row, mu_col: str, background_dict: dict):

    # time
    t = row['Time (h)']

    # mu_bg 
    mu_bg = background_dict[t][mu_col]

    return row[mu_col] / mu_bg