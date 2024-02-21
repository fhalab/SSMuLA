
"""
A script for processing zs data
"""

import pandas as pd
from Bio import SeqIO

from SSMuLA.landscape_global import LIB_INFO_DICT

class DataProcessor:
    def __init__(self):
        pass

    def get_Meta(self, df):
        """Get metadata from the data summary csv"""
        # Check if end is xlsx or csv
        if df.endswith(".xlsx"):
            data = pd.read_excel(df, sheet_name=1)
        elif df.endswith(".csv"):
            data = pd.read_csv(df)
        
        # append mut
        return self._append_mut(data)
    

    def _convert_muts(self, muts: str) -> str:

        """
        Convert the variants sequence
        to the form of parentaa1loc1mutaa1:parentaa2loc2mutaa2
        """

        mut_seq = ""
        mut_num = 0

        for i, (mut, wt) in enumerate(
            zip(muts, LIB_INFO_DICT["AAs"].values())
        ):
            if mut != wt:
                mut_num += 1
                if mut_num != 1:
                    mut_seq += ":" + LIB_INFO_DICT["AAs"][i+1] + mut
                else:
                    mut_seq += LIB_INFO_DICT["AAs"][i+1] + mut
        return mut_seq
    
    def _append_mut(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Apply the convert_muts function to the dataframe
        """

        df_appended = df.copy()
        df_appended.loc[:, "muts"] = df_appended.apply(
            lambda x: self._convert_muts(x["fitness"]),
            axis=1,
        )

        return df_appended.replace("", "WT")

    def get_Seq(self, path):
        return SeqIO.read(path, "fasta").seq

    def get_mut(self, df):
        return df["muts"].values

    def count_mut(self, df):
        """Count the number of mutations in the data set"""
        muts = self.get_mut(df)
        muts = [m.split(":") for m in muts]
        muts = [len(m) for m in muts]
        return muts

    def split_input(self, input_string):
        if (
            len(input_string) >= 3
            and input_string[0].isalpha()
            and input_string[-1].isalpha()
        ):
            wt_AA = input_string[0]
            new_AA = input_string[-1]
            pos = input_string[1:-1]

            try:
                pos = int(pos)  # Convert the position string to an integer
            except ValueError:
                return "Wrong input format. Please use the format '[WTAA][Pos][NewAA]'"

            return wt_AA, pos, new_AA
        else:
            # Assign NA to three variables
            wt_AA, pos, new_AA = "NA", "NA", "NA"
            return wt_AA, pos, new_AA

    def get_unique_mut_pos(self, df):
        """Get the unique position from the list of mutations to mask certain regions"""
        muts = self.get_mut(df)
        muts = [m.split(":") for m in muts]
        muts = [m for sublist in muts for m in sublist]
        muts = [self.split_input(m) for m in muts]
        muts = [m[1] for m in muts]
        muts = list(set(muts))
        muts = [m for m in muts if m != "NA"]
        return muts

    def replace_aa(self, seq, pos, new_aa):
        """Replace amino acid in a given position"""
        seq = list(seq)
        seq[pos - 1] = new_aa
        return "".join(seq)

    def mutate_single(self, wt, mut):
        """Mutate a given sequence"""

        mut_seq = wt
        if ":" in mut:
            mut = mut.split(":")
            for m in mut:
                m = self.split_input(m)
                mut_seq = self.replace_aa(mut_seq, m[1], m[2])
        else:
            m = self.split_input(mut)
            mut_seq = self.replace_aa(mut_seq, m[1], m[2])
        return mut_seq

    def get_pos(self, mut):
        if ":" in mut:
            mut = mut.split(":")
            pos = [
                self.split_input(m)[1] for m in mut
            ]  # Extract positions for each mutation
            pos = [int(p) for p in pos]  # Convert positions to integers
        elif mut == "WT":
            pos = "NA"
        else:
            m = self.split_input(mut)
            pos = m[1]
            pos = [int(pos)]
        return pos

    def store_pos_list(self, df, append=True):
        pos_list = []
        for mut in df["muts"]:
            pos = self.get_pos(mut)
            if pos != "NA":
                pos_list.append(pos)

            elif pos == "NA":
                pos_list.append("NA")

        if append:
            series = pd.Series(self.store_pos_list(df, append=False))
            df["pos"] = series
            return df

        else:
            return pos_list

    def get_mutated_aa(self, mut):
        if ":" in mut:
            mut = mut.split(":")
            aa = [m[2] for m in mut]
        else:
            m = self.split_input(mut)
            aa = m[2]
        return aa

    def get_combo(self, mut, list=False):
        """
        Create sequential AA combo of the mutated sequences
        
        Input: Mut Variants [WT][Pos][NewAA]]
        Output: Sequential AA combo
        """
        combo = []
        if ":" in mut:
            mut = mut.split(":")
            for m in mut:
                m = self.split_input(m)
                combo.append(m[2])
        else:
            m = self.split_input(mut)
            combo.append(m[2])

        if list:
            return combo
        else:
            return "".join(combo)

    def mutate_all(self, df, wt, _combo=False):
        """Mutate all sequences"""
        muts = self.get_mut(df)
        new_seqs = []
        Combo = []
        for mut in muts:

            if _combo:
                Combo.append(self.get_combo(mut, list=True))

            if mut == "WT":
                new_seqs.append(wt)
                continue

            seq = self.mutate_single(wt, mut)

            if seq != "NA":
                new_seqs.append(seq)
            else:
                new_seqs.append("NA")

        df["seq"] = new_seqs

        if _combo:
            df["combo"] = Combo

        return df

    def prepare_zero_shot(self, path, fasta, _combo=False, _pos=True):
        """
        Extract the sequences and prepare the mutations for zero-shot prediction

        Input:
            path: path to the data summary csv of the data set
            fasta: path to the fasta file of the data set (wt sequence)

        Output:
            df: dataframe with the sequences and the mutations
        """

        data = pd.read_csv(path)

        # Try to drop column "Unamed:0"
        try:
            data = data.drop("Unnamed: 0", axis=1)
        except KeyError:
            pass

        wt = self.get_Seq(fasta)
        data = self.mutate_all(data, wt, _combo=_combo)

        if _pos:
            data = self.store_pos_list(data)
        return data
