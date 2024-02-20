# general imports
import pandas as pd
import numpy as np
import typing

# imports for parsing fastq files
from Bio import Seq,SeqIO
import pysam
pysam.set_verbosity(0)

import gzip
import datetime
import multiprocessing
import itertools
import pathlib
from tqdm import tqdm
import glob
from .globals import *

#################### GENERAL LIBRARY CLASS ###################
class Library:
    def __init__(
            self,
            library: str
            ):
        # save inputs as attributues
        self.library = library

        # obtain overall globals
        self.parent_DNA_seq = TM9D8S_DNA
        self.parent_AA_seq = TM9D8S_AA

        # obtain global variables for a given library
        self.positions = LIBRARY_INFO_DICT[self.library]['positions']
        self.parent_codons = LIBRARY_INFO_DICT[self.library]['codons']
        self.parent_AAs = LIBRARY_INFO_DICT[self.library]['AAs']
        self.OD_dict = OD_DICT[self.library]


#################### Process SAM files ####################

class SamFileParser(Library):
    def __init__(
            self, 
            library: str,
            sequencing_date: str, 
            n_positions: int,
            base_folder: str,
            sam_folder: str,
            output_folder: str=None
            ):
        
        super().__init__(library)

        # save inputs as attributues
        self.n_positions = n_positions
        self.base_folder = base_folder
        self.sam_folder = sam_folder
        self.sequencing_date = sequencing_date

        # calculate some new attributes based on inputs
        self.current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_folder is None:
            self.output_folder = f'{self.base_folder}Lib{self.library}_results/{self.current_date}/'
        else:
            self.output_folder = output_folder

        # parsing specific attributes
        self.file_mapping_dict = SAMFILE_MAPPING[self.library][self.sequencing_date]
        self.codon_starts = SAMFILE_CODONS[library]

        # check that the number of positions is correct
        if len(list(self.codon_starts['fwd'].keys()) + list(self.codon_starts['rev'].keys())) != self.n_positions:
            raise ValueError
        
    def get_read_df(self, rep_key):
        """
        Function that takes a directory as well as a dictionary with forward
        and reverse files names and a dictionary with information about codon
        start and returns dictionaries of fwd and reverse reads for the given
        file pair and codons of interest.
        """

        files = glob.glob(self.sam_folder+self.file_mapping_dict[rep_key])

        # check that files is only one file with an assert statement
        assert len(files) == 1, f'Error: more than one file found for replicate {rep_key}!'

        sam_file = files[0]
        alignment = pysam.AlignmentFile(sam_file, "rb")
        reads = alignment.fetch(until_eof=True)

        # make dictionary to store reads
        print(f'Parsing {rep_key}...')

        read_dictionary = {}
        for read in reads:

            seq_id = str(read.query_name)

            # check if the ID already exists in the dictionary
            if seq_id not in read_dictionary:
                read_dictionary[seq_id] = {}

            seq = str(read.seq)

            # even numbers are forward reads
            if read.is_forward:
                
                read_dictionary[seq_id]['fwd'] = seq

                for codon,base_start in self.codon_starts['fwd'].items():

                    codon_seq = seq[base_start:base_start+3]
                    # AA_seq = TRANSLATE_DICT[codon_seq]

                    read_dictionary[seq_id][codon] = codon_seq
                    # read_dictionary[seq_id][f'AA{codon[-1]}'] = AA_seq


            # odd numbers are reverse reads
            elif read.is_reverse:

                read_dictionary[seq_id]['rev'] = seq
                
                for codon,base_start in self.codon_starts['rev'].items():
                    
                    codon_seq = seq[base_start:base_start+3]
                    # AA_seq = TRANSLATE_DICT[codon_seq]

                    read_dictionary[seq_id][codon] = codon_seq
                    # read_dictionary[seq_id][f'AA{codon[-1]}'] = AA_seq

            else:
                raise ValueError(f'Error: read {seq_id} is neither forward nor reverse!')

        print(f'Finished parsing {rep_key}! Now converting to DataFrame...')
        paired_reads = pd.DataFrame(read_dictionary).T.reset_index(drop=True)

        paired_reads = paired_reads[paired_reads['fwd'].notna() & paired_reads['rev'].notna()].copy().reset_index(drop=True)

        return paired_reads
        
    
    def process_replicate(self,rep_key):
        """
        Function that takes a replicate key and returns a dictionary of codons and AAs DataFrames. Ready to be multiprocessed!
        """

        paired_read_df = self.get_read_df(rep_key)

        print(f'Finished converting {rep_key} to DataFrame! Now processing codons...')

        ### Get codon counts and write a codon column ###
        codon_counts = paired_read_df.groupby([f'codon{i}' for i in range(1,self.n_positions+1)]).size().reset_index().rename(columns={0:'count'})

        codon_counts['codons'] = codon_counts.apply(lambda row: '_'.join([row[f'codon{i}'] for i in range(1,self.n_positions+1)]), axis=1)

        ### Get AA counts and write an AA column ###
        # AA_counts = paired_read_df.groupby([f'AA{i}' for i in range(1,5)]).size().reset_index().rename(columns={0:'count'})

        # AA_counts['AAs'] = AA_counts.apply(lambda row: '_'.join([row[f'AA{i}'] for i in range(1,self.n_positions+1)]), axis=1)

        ### Deleted paired read df to save memory ###
        del paired_read_df

        return codon_counts


    def run_parsing(self, n_jobs=None):
        """
        """

        print('Initiating parsing...')

        # write output folder
        pathlib.Path(self.output_folder).mkdir(parents=True,exist_ok=True)

        time_rep_keys = list(self.file_mapping_dict.keys())
        # time_rep_keys = ['T0_rep1', 'T0_rep2', 'T3_rep1', 'T3_rep2']
        print(time_rep_keys)
        
        print('Processing replicates with multiprocessing...')
        if n_jobs is None:
            with multiprocessing.Pool(processes=len(time_rep_keys)) as pool:
                result = pool.map(self.process_replicate, time_rep_keys)
        else:
            with multiprocessing.Pool(processes=n_jobs) as pool:
                result = pool.map(self.process_replicate, time_rep_keys)

        self.codon_count_dfs = {time_rep_keys[i]: result[i] for i in range(len(result))}

        # self.AA_count_dfs = {time_rep_keys[i]: result[i]['AAs'] for i in range(len(result))}

        self.write_count_files()

    def write_count_files(self):
        """
        """

        # get the date and time
        print('Writing codon files...')
        # write codon files for the given timepoint
        for rep_key, codon_counts in self.codon_count_dfs.items():
            file_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            codon_counts.to_csv(f'{self.output_folder}Lib{self.library}_{rep_key}_{file_date}_codons.csv')

        # print('Writing AA files...')
        # # write AA files for the given timepoint
        # for rep_key, AA_counts in self.AA_count_dfs.items():
        #     file_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        #     AA_counts.to_csv(f'{self.output_folder}Lib{self.library}_{rep_key}_{file_date}_AAs.csv')

##################### COMBINE TIMEPOINTS #####################
class CombineTimepoints(Library):
    def __init__(
            self, 
            library: str,
            processed_data_folder: str,
            n_positions: int,
            drop_non_NNK: bool=True
            ):
        
        super().__init__(library)

        self.processed_data_folder = processed_data_folder
        self.n_positions = n_positions
        self.drop_non_NNK = drop_non_NNK

        self.timepoint_dict = TIMEPOINT_DICT[self.library]
        
        # populate the file dictionary
        self.populate_file_dict()

        # get dictionaries for AAs and codons DataFrames
        self.get_codon_dfs()

        # get merged DataFrames for AAs and codons
        self.merged_codons = self.merge_timepoint_dfs()
        self.get_merged_AAs()

    def populate_file_dict(self):
        
        self.timepoint_files = {'rep1': {}, 'rep2': {}}

        for timepoint in self.timepoint_dict.keys():
            self.timepoint_files['rep1'][timepoint] = {'codons': {}}
            self.timepoint_files['rep2'][timepoint] = {'codons': {}}

        # Get the file paths with pathlib
        file_paths = pathlib.Path(self.processed_data_folder).glob('*.csv')

        # loop through all of the globbed paths
        for file_path in file_paths:

            file = file_path.stem
            _, timepoint, replicate, _, _, seq_type = tuple(file.split('_'))
            self.timepoint_files[replicate][timepoint][seq_type] = file_path

    def get_codon_dfs(self):

        # Get the codon CSV files from 
        codon_dfs_dict = {}

        for replicate in self.timepoint_files.keys():
            codon_dfs_dict[replicate] = {}
            
            for timepoint in self.timepoint_files[replicate].keys():
                try: 
                    temp = pd.read_csv(
                        self.timepoint_files[replicate][timepoint]['codons'], 
                        index_col=0
                        )

                    codon_dfs_dict[replicate][timepoint] = temp.copy()
                
                except:
                    print(f'No data found for {replicate}, {timepoint}')
                
        self.codon_dfs_dict = codon_dfs_dict

    def drop_non_NNK_codons(self, df):
        """Takes in a dictionary of codons and drops all rows that do not have a T or G (K codon) in the third position. It also records how many actual sequences are dropped for each dataframe.

        Args:
            df (pd.DataFrame): DataFrame with columns codon1, codon2, ..., and count

        Returns:
            df (pd.DataFrame): Input DataFrame but with NNK codons dropped
        """
        def check_accepted(row):

            # Check if any of the codons is not a T or G 
            for i in range(1,self.n_positions+1):

                # If any codon doesn't have T or G return False
                if row[f'codon{i}'][-1] not in ['T', 'G']:
                    return False
                
            # Otherwise, the codon passes!
            return True
        
        mask = df.apply(check_accepted, axis=1)

        # Return the total count of sequences dropped
        dropped = df[~mask]['count'].sum()
        total = df['count'].sum()
        print(f'Dropped {dropped} sequences from {total} total sequences because they were non-NNK codons\n')

        df = df[mask].copy()

        return df

    def process_rep_data(self, T0_df):
        """
        Returns a combined dataframe of all timepoints in a replicate
        """

        replicates = {}
        for rep,rep_dict in self.codon_dfs_dict.items():
            
            df_list = []

            for time,df in rep_dict.items():
                
                if time != 'T0':
                    print(f'Replicate {rep}, timepoint {time}:')

                    if self.drop_non_NNK:
                        df = self.drop_non_NNK_codons(df).copy()

                    _df = pd.merge(
                        T0_df.copy(), 
                        df, 
                        on=[f'codon{i}' for i in range(1,self.n_positions+1)]+['codons'],
                        how='outer'
                        )
                    _df = _df.rename(columns={'count': 'OutputCount'})
                    _df['OutputFreq'] = _df['OutputCount'] / _df['OutputCount'].sum()
                    _df['Timepoint'] = time
                    df_list.append(_df)
            
            if len(df_list) != 0:
                replicates[rep] = pd.concat(df_list).reset_index(drop=True)
        
        for rep,df in replicates.items():
            
            if rep != 'rep1':
                merged_reps = pd.merge(
                    merged_reps, 
                    df, 
                    on=list(T0_df.columns)+['Timepoint'],
                    how='outer',
                    suffixes=('_1','_2')
                    ).copy()
            else:
                merged_reps = df.copy()

        merged_reps['Time (h)'] = merged_reps['Timepoint'].apply(lambda x: self.timepoint_dict[x])

        old_cols = list(merged_reps.columns)
        old_cols.remove('Timepoint')
        old_cols.remove('Time (h)')
        new_cols = ['Timepoint', 'Time (h)'] + old_cols
        
        merged_reps = merged_reps[new_cols].copy()
        
        return merged_reps

    def merge_timepoint_dfs(self):
        """
        - Takes a dictionary of codon_dfs
        - Returns a DataFrame of merged replicates/timepoints for a given library/seq_type
        """

        # Get T0 data from all replicates with T0 data

        T0_dfs = []
        for replicate in self.codon_dfs_dict.keys():
            for timepoint,df in self.codon_dfs_dict[replicate].items():
                if timepoint == 'T0':
                    print(f'Replicate {replicate}, timepoint {timepoint}:')
                    
                    if self.drop_non_NNK:
                        df = self.drop_non_NNK_codons(df).copy()

                    T0_dfs.append(df)
        
        assert len(T0_dfs) > 0, 'No T0 data found for any replicate'
        
        T0_df = T0_dfs[0].copy()

        if len(T0_dfs) > 1:
            for i in range(1, len(T0_dfs)):
                T0_df = pd.merge(
                    left=T0_df, 
                    right=T0_dfs[i],
                    how='outer',
                    on=[f'codons']+[f'codon{n}' for n in range(1,self.n_positions+1)], 
                    suffixes=[f'_{n}' for n in range(1,len(T0_dfs)+1)]
                    )
        else:
            T0_df = T0_df.rename(columns={'count': 'count_1'})
        
        T0_df = T0_df.reindex(sorted(T0_df.columns), axis=1)

        T0_df = T0_df.rename(columns={f'count_{i}': f'InputCount_{i}' for i in range(1, len(T0_dfs)+1)})

        T0_df['avg_InputCount'] = T0_df[[f'InputCount_{i}' for i in range(1, len(T0_dfs)+1)]].mean(axis=1)

        for i in range(1, len(T0_dfs)+1):
            count_sum = T0_df[f'InputCount_{i}'].sum()
            T0_df[f'InputFreq_{i}'] = T0_df[f'InputCount_{i}'] / count_sum

        T0_df['avg_InputFreq'] = T0_df[[f'InputFreq_{i}' for i in range(1, len(T0_dfs)+1)]].mean(axis=1)
        
        merged_timepoints = self.process_rep_data(
                T0_df=T0_df,
            )

        return merged_timepoints
    
    def get_merged_AAs(self):

        merged_AAs = self.merged_codons.copy()

        # Filter out any rows that have NaNs in either InputCount column
        merged_AAs = merged_AAs.dropna(subset=['InputCount_1'])

        try:
            merged_AAs = merged_AAs.dropna(subset=['InputCount_2'])
        except:
            print('No InputCount_2 column found')

        for i in range(1, self.n_positions+1):
            merged_AAs.insert(1+i, f'AA{i}', merged_AAs[f'codon{i}'].apply(lambda x: TRANSLATE_DICT[x]))

        # Drop the codon columns
        merged_AAs.drop(columns=[f'codon{i}' for i in range(1, self.n_positions+1)]+['codons'], inplace=True)

        # Group by timepoint, time, and AA1-AA4 and sum the counts
        merged_AAs = merged_AAs.groupby(['Timepoint', 'Time (h)'] + [f'AA{i}' for i in range(1, self.n_positions+1)]).sum().reset_index()

        merged_AAs.insert(2, 'AAs', merged_AAs.apply(lambda row: ''.join([row[f'AA{i}'] for i in range(1, self.n_positions+1)]), axis=1))

        self.merged_AAs = merged_AAs.copy()

    
    # def write_merged_data(self):

    #     file_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     output_folder = f'{self.processed_data_folder}{file_date}_merged/'

    #     pathlib.Path(output_folder).mkdir(parents=True,exist_ok=True)

    #     self.merged_codons.to_csv(f'{output_folder}{file_date}_merged_codons.csv')


   





#### ARCHIVED FUNCTIONS ####

# class CombineTimepoints(Library):
#     def __init__(
#             self, 
#             library: str,
#             processed_data_folder: str,
#             n_positions: int,
#             ):
        
#         super().__init__(library)

#         self.processed_data_folder = processed_data_folder
#         self.n_positions = n_positions

#         self.timepoint_dict = TIMEPOINT_DICT[self.library]
        
#         # populate the file dictionary
#         self.populate_file_dict()

#         # get dictionaries for AAs and codons DataFrames
#         self.get_AA_dfs()
#         self.get_codon_dfs()

#         # get merged DataFrames for AAs and codons
#         self.merged_AAs = self.merge_timepoint_dfs(seq_type='AA')
#         self.merged_codons = self.merge_timepoint_dfs(seq_type='codon')

#     def populate_file_dict(self):
        
#         self.timepoint_files = {'rep1': {}, 'rep2': {}}

#         for timepoint in self.timepoint_dict.keys():
#             self.timepoint_files['rep1'][timepoint] = {'codons': {}, 'AAs': {}}
#             self.timepoint_files['rep2'][timepoint] = {'codons': {}, 'AAs': {}}

#         # Get the file paths with pathlib
#         file_paths = pathlib.Path(self.processed_data_folder).glob('*.csv')

#         # loop through all of the globbed paths
#         for file_path in file_paths:

#             file = file_path.stem
#             _, timepoint, replicate, _, _, seq_type = tuple(file.split('_'))
#             self.timepoint_files[replicate][timepoint][seq_type] = file_path

#     def get_codon_dfs(self):

#         # Get the codon CSV files from 
#         codon_dfs_dict = {}

#         for replicate in self.timepoint_files.keys():
#             codon_dfs_dict[replicate] = {}
            
#             for timepoint in self.timepoint_files[replicate].keys():
#                 temp = pd.read_csv(
#                     self.timepoint_files[replicate][timepoint]['codons'], 
#                     index_col=0
#                     )

#                 codon_dfs_dict[replicate][timepoint] = temp.copy()
                
#         self.codon_dfs_dict = codon_dfs_dict

#     def get_AA_dfs(self):

#         AA_dfs_dict = {}

#         for replicate in self.timepoint_files.keys():
#             AA_dfs_dict[replicate] = {}
            
#             for timepoint in self.timepoint_files[replicate].keys():
#                 temp = pd.read_csv(
#                     self.timepoint_files[replicate][timepoint]['AAs'], 
#                     index_col=0
#                     )

#                 AA_dfs_dict[replicate][timepoint] = temp.copy()

#         self.AA_dfs_dict = AA_dfs_dict

#     def process_rep_data(self, T0_df, seq_type):
#         """
#         Returns a combined dataframe of all timepoints in a replicate
#         """

#         replicates = {}

#         if seq_type == 'AA':
#             for rep,rep_dict in self.AA_dfs_dict.items():
               
#                 df_list = []

#                 for time,df in rep_dict.items():
#                     if time != 'T0':
#                         _df = pd.merge(
#                             T0_df.copy(), 
#                             df, 
#                             on=[f'AA{i}' for i in range(1,self.n_positions+1)]+['AAs'],
#                             how='outer'
#                             )
#                         _df = _df.rename(columns={'count': 'OutputCount'})
#                         _df['OutputFreq'] = _df['OutputCount'] / _df['OutputCount'].sum()
#                         _df['Timepoint'] = time
#                         df_list.append(_df)
                
#                 replicates[rep] = pd.concat(df_list).reset_index(drop=True)

#         elif seq_type == 'codon':
#             for rep,rep_dict in self.codon_dfs_dict.items():
               
#                 df_list = []

#                 for time,df in rep_dict.items():
#                     if time != 'T0':
#                         _df = pd.merge(
#                             T0_df.copy(), 
#                             df, 
#                             on=[f'codon{i}' for i in range(1,self.n_positions+1)]+['codons'],
#                             how='outer'
#                             )
#                         _df = _df.rename(columns={'count': 'OutputCount'})
#                         _df['OutputFreq'] = _df['OutputCount'] / _df['OutputCount'].sum()
#                         _df['Timepoint'] = time
#                         df_list.append(_df)
                
#                 replicates[rep] = pd.concat(df_list).reset_index(drop=True)
        
#         for rep,df in replicates.items():
            
#             if rep != 'rep1':
#                 merged_reps = pd.merge(
#                     merged_reps, 
#                     df, 
#                     on=list(T0_df.columns)+['Timepoint'],
#                     how='outer',
#                     suffixes=('_1','_2')
#                     ).copy()
#             else:
#                 merged_reps = df.copy()

#         merged_reps['Time (h)'] = merged_reps['Timepoint'].apply(lambda x: self.timepoint_dict[x])

#         old_cols = list(merged_reps.columns)
#         old_cols.remove('Timepoint')
#         old_cols.remove('Time (h)')
#         new_cols = ['Timepoint', 'Time (h)'] + old_cols
        
#         merged_reps = merged_reps[new_cols].copy()
        
#         return merged_reps

#     def merge_timepoint_dfs(self, seq_type):
#         """
#         - Takes either a dictionary of codon_dfs or AA_dfs
#         - Returns a DataFrame of merged replicates/timepoints for a given library/seq_type
#         """

#         # Get T0 data from all replicates with T0 data
#         if seq_type == 'AA':
#             T0_dfs = []
#             for replicate in self.AA_dfs_dict.keys():
#                 for timepoint in self.AA_dfs_dict[replicate].keys():
#                     if timepoint == 'T0':
#                         T0_dfs.append(self.AA_dfs_dict[replicate][timepoint])
            
#             assert len(T0_dfs) > 0, 'No T0 data found for any replicate'

#         elif seq_type == 'codon':
#             T0_dfs = []
#             for replicate in self.codon_dfs_dict.keys():
#                 for timepoint in self.codon_dfs_dict[replicate].keys():
#                     if timepoint == 'T0':
#                         T0_dfs.append(self.codon_dfs_dict[replicate][timepoint])
            
#             assert len(T0_dfs) > 0, 'No T0 data found for any replicate'

#         else:
#             raise ValueError('seq_type must be either "AA" or "codon"')
        
#         T0_df = T0_dfs[0].copy()

#         for i in range(1, len(T0_dfs)):
#             T0_df = pd.merge(
#                 left=T0_df, 
#                 right=T0_dfs[i],
#                 how='outer',
#                 on=[f'{seq_type}s']+[f'{seq_type}{n}' for n in range(1,self.n_positions+1)], 
#                 suffixes=[f'_{n}' for n in range(1,len(T0_dfs)+1)]
#                 )
        
#         T0_df = T0_df.reindex(sorted(T0_df.columns), axis=1)

#         T0_df = T0_df.rename(columns={f'count_{i}': f'InputCount_{i}' for i in range(1, len(T0_df)+1)})

#         T0_df['avg_InputCount'] = T0_df[[f'InputCount_{i}' for i in range(1, len(T0_dfs)+1)]].mean(axis=1)

#         for i in range(1, len(T0_dfs)+1):
#             count_sum = T0_df[f'InputCount_{i}'].sum()
#             T0_df[f'InputFreq_{i}'] = T0_df[f'InputCount_{i}'] / count_sum

#         T0_df['avg_InputFreq'] = T0_df[[f'InputFreq_{i}' for i in range(1, len(T0_dfs)+1)]].mean(axis=1)

#         if seq_type == 'AA': 
#             merged_timepoints = self.process_rep_data(
#                     T0_df=T0_df,
#                     seq_type=seq_type
#                 )
#         elif seq_type == 'codon':
#             merged_timepoints = self.process_rep_data(
#                 T0_df=T0_df,
#                 seq_type=seq_type
#             )

#         return merged_timepoints
    
#     def write_merged_data(self):

#         file_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#         output_folder = f'{self.processed_data_folder}{file_date}_merged/'

#         pathlib.Path(output_folder).mkdir(parents=True,exist_ok=True)

#         self.merged_AAs.to_csv(f'{output_folder}{file_date}_merged_AAs.csv')


###################### FAST-PARSE FASTQ ######################
# class FastqParser(Library):
#     def __init__(
#             self, 
#             library: str,
#             sequencing_date: str, 
#             n_positions: int,
#             base_folder: str,
#             fastq_folder: str,
#             output_folder: str=None
#             ):
        
#         super().__init__(library)

#         # save inputs as attributues
#         self.n_positions = n_positions
#         self.base_folder = base_folder
#         self.fastq_folder = fastq_folder
#         self.sequencing_date = sequencing_date

#         # calculate some new attributes based on inputs
#         self.current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
#         if output_folder is None:
#             self.output_folder = f'{self.base_folder}Lib{self.library}_results/{self.current_date}/'
#         else:
#             self.output_folder = output_folder

#         # parsing specific attributes
#         self.fwd_ref_read = REFERENCE_READS[self.library]['fwd']
#         self.rev_ref_read = REFERENCE_READS[self.library]['rev'] 
#         self.file_mapping_dict = FASTQ_MAPPING[self.library][self.sequencing_date]
#         self.codon_starts = CODON_STARTS[library]

#         # check that the number of positions is correct
#         if len(list(self.codon_starts['fwd'].keys()) + list(self.codon_starts['rev'].keys())) != self.n_positions:
#             raise ValueError
        
#     def _check_parent(self):
        
#         parent_dict = {
#             'codons': {},
#             'AAs': {}
#         }

#         print('forward:')
#         for codon_num, position in self.codon_starts['fwd'].items():

#             codon = Seq.Seq(self.fwd_ref_read[position-1:position+2])
#             AA = codon.translate()
#             parent_dict['codons'][codon_num] = str(codon)
#             parent_dict['AAs'][codon_num] = str(AA)
#             print(f'{codon_num} starts at forward nt {position} with bases {codon} that translate to {AA}')

#         print('\nreverse:')
#         for codon_num, position in self.codon_starts['rev'].items():

#             codon = Seq.Seq(self.rev_ref_read[position-1:position+2]).reverse_complement()
#             AA = codon.translate()
#             parent_dict['codons'][codon_num] = str(codon)
#             parent_dict['AAs'][codon_num] = str(AA)
#             print(f'{codon_num} starts at reverse nt {position} with bases {codon} that translate to {AA}')

#         print(f'\nLibrary {self.library} parent dictionary:\n{parent_dict}')


#     def get_read_df(self, file_pair_dict):
#         """
#         Function that takes a directory as well as a dictionary with forward
#         and reverse files names and a dictionary with information about codon
#         start and returns dictionaries of fwd and reverse reads for the given
#         file pair and codons of interest.
#         """
#         fwd_file = glob.glob(self.fastq_folder+file_pair_dict['fwd_file'])[0]
#         rev_file = glob.glob(self.fastq_folder+file_pair_dict['rev_file'])[0]

#         # Populate dictionary with fwd reads
#         fwd_read_dict = {}
#         with gzip.open(fwd_file, "rt") as handle:
#             for record in SeqIO.parse(handle, "fastq"):
#                 fwd_read_dict[record.id] = str(record.seq)

#         # Populate dictionary with rev reads
#         rev_read_dict = {}
#         with gzip.open(rev_file, "rt") as handle:
#             for record in SeqIO.parse(handle, "fastq"):
#                 rev_read_dict[record.id] = str(record.seq)

#         paired_read_df = pd.concat([pd.Series(fwd_read_dict, name='fwd'), pd.Series(rev_read_dict, name='rev')], axis=1).reset_index(drop=True)

#         del fwd_read_dict, rev_read_dict

#         return paired_read_df
    
#     def process_replicate(self,rep_key):
#         """
#         Function that takes a replicate key and returns a dictionary of codons and AAs DataFrames. Ready to be multiprocessed!
#         """
       
#         file_pair_dict = self.file_mapping_dict[rep_key]
        
#         paired_read_df = self.get_read_df(file_pair_dict)

#         # loop through the positions and get the codons and AAs
#         for codon,position in self.codon_starts['fwd'].items():
            
#             # Get codon and AA for given position
#             paired_read_df[codon] = paired_read_df['fwd'].apply(lambda x: x[position-1:position+2])

#             paired_read_df[f'AA{codon[-1]}'] = paired_read_df[codon].apply(lambda x: Seq.Seq(x).translate()[0])

#         for codon,position in self.codon_starts['rev'].items():

#             # Get codon and AA for given position
#             paired_read_df[codon] = paired_read_df['rev'].apply(lambda x: str(Seq.Seq(x[position-1:position+2]).reverse_complement()))

#             paired_read_df[f'AA{codon[-1]}'] = paired_read_df[codon].apply(lambda x: Seq.Seq(x).translate()[0])

#         paired_read_df['codons'] = paired_read_df.apply(lambda row: '_'.join([row[f'codon{i}'] for i in range(1,self.n_positions+1)]), axis=1)

#         paired_read_df['AAs'] = paired_read_df.apply(lambda row: '_'.join([row[f'AA{i}'] for i in range(1,self.n_positions+1)]), axis=1)

#         # convert these into codon and AA dictionaries for downstream processing
#         # initalize them with 0 for each codon combination!

#         codon_counts = pd.DataFrame(paired_read_df.value_counts(['codons']), columns=['count']).sort_values('codons').reset_index()

#          # get counts of codons and AAs as dataframes
#         AA_counts = pd.DataFrame(paired_read_df.value_counts(['AAs']), columns=['count']).sort_values('AAs').reset_index()

#         # Deleted paired read df to save memory
#         del paired_read_df

#         for i in range(1,1+self.n_positions):
#             AA_counts[f'AA{i}'] = AA_counts['AAs'].apply(lambda x: x.split('_')[i-1])
#             codon_counts[f'codon{i}'] = codon_counts['codons'].apply(lambda x: x.split('_')[i-1])

#         return {'codons': codon_counts, 'AAs': AA_counts}

#     def run_parsing(self, n_jobs=None):
#         """
#         """

#         print('Initiating parsing...')

#         # write output folder
#         pathlib.Path(self.output_folder).mkdir(parents=True,exist_ok=True)

#         time_rep_keys = list(self.file_mapping_dict.keys())
#         print(time_rep_keys)
        
#         print('Processing replicates with multiprocessing...')
#         if n_jobs is None:
#             with multiprocessing.Pool(processes=len(time_rep_keys)) as pool:
#                 result = pool.map(self.process_replicate, time_rep_keys)
#         else:
#             with multiprocessing.Pool(processes=n_jobs) as pool:
#                 result = pool.map(self.process_replicate, time_rep_keys)

#         self.codon_count_dfs = {time_rep_keys[i]: result[i]['codons'] for i in range(len(result))}

#         self.AA_count_dfs = {time_rep_keys[i]: result[i]['AAs'] for i in range(len(result))}

#         self.write_count_files()

#     def write_count_files(self):
#         """
#         """

#         # get the date and time
#         print('Writing codon files...')
#         # write codon files for the given timepoint
#         for rep_key, codon_counts in self.codon_count_dfs.items():
#             file_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#             codon_counts.to_csv(f'{self.output_folder}Lib{self.library}_{rep_key}_{file_date}_codons.csv')

#         print('Writing AA files...')
#         # write AA files for the given timepoint
#         for rep_key, AA_counts in self.AA_count_dfs.items():
#             file_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#             AA_counts.to_csv(f'{self.output_folder}Lib{self.library}_{rep_key}_{file_date}_AAs.csv')