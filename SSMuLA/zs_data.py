"""
A script for processing zs data
"""

import warnings
import os
from copy import deepcopy
import pandas as pd

from Bio import SeqIO, pairwise2, PDB

from SSMuLA.landscape_global import LIB_INFO_DICT

warnings.filterwarnings("ignore")


EV_META = {
    "DHFR": {
        "recommended": {
            "bitscore": 0.7,
            "sequences": 16042,
            "seqs_per_l": 103.5,
            "quality": 10,
        },
        "other_1": {
            "bitscore": 0.1,
            "sequences": 59494,
            "seqs_per_l": 491.7,
            "quality": 10,
        },
        "other_2": {
            "bitscore": 0.3,
            "sequences": 17758,
            "seqs_per_l": 114.6,
            "quality": 10,
        },
        "other_3": {
            "bitscore": 0.5,
            "sequences": 17132,
            "seqs_per_l": 111.2,
            "quality": 10,
        },
    },
    "ParD2": {
        "recommended": {
            "bitscore": 0.3,
            "sequences": 13911,
            "seqs_per_l": 252.9,
            "quality": 10,
        },
        "other_1": {
            "bitscore": 0.1,
            "sequences": 102977,
            "seqs_per_l": 2060,
            "quality": 9,
        },
        "chosen": {
            "bitscore": 0.5,
            "sequences": 6789,
            "seqs_per_l": 93.0,
            "quality": 9,
        },
        "other_3": {
            "bitscore": 0.7,
            "sequences": 3638,
            "seqs_per_l": 49.2,
            "quality": 8,
        },
    },
    "ParD3": {
        "recommended": {
            "bitscore": 0.3,
            "sequences": 13189,
            "seqs_per_l": 235.5,
            "quality": 10,
        },
        "other_1": {
            "bitscore": 0.1,
            "sequences": 117463,
            "seqs_per_l": 2303,
            "quality": 9,
        },
        "chosen": {
            "bitscore": 0.5,
            "sequences": 6784,
            "seqs_per_l": 91.7,
            "quality": 9,
        },
        "other_3": {
            "bitscore": 0.7,
            "sequences": 3249,
            "seqs_per_l": 43.3,
            "quality": 8,
        },
    },
    "GB1": {
        "recommended": {
            "bitscore": 0.3,
            "sequences": 4358,
            "seqs_per_l": 128.2,
            "quality": 8,
        },
        "other_1": {
            "bitscore": 0.1,
            "sequences": 22512,
            "seqs_per_l": 750.4,
            "quality": 8,
        },
        "chosen": {
            "bitscore": 0.5,
            "sequences": 29,
            "seqs_per_l": 0.5,
            "quality": 3,
        },
        "other_3": {
            "bitscore": 0.7,
            "sequences": 27,
            "seqs_per_l": 0.5,
            "quality": 3,
        },
    },
    "T7": {
        "chosen": {
            "bitscore": 0.95,
            "sequences": 309,
            "seqs_per_l": 0.7,
            "quality": 6,
            "position filter": 0.45
        },
    },
    "TEV": {
        # "chosen": {
        #     "bitscore": 0.1,
        #     "sequences": 163169,
        #     "seqs_per_l": 1073,
        #     "quality": 10,
        # },
        "chosen": {
            "bitscore": 0.7,
            "sequences": 164,
            "seqs_per_l": 0.7,
            "quality": 6,
        },
    },
    "TrpB": {
        "recommended": {
            "bitscore": 0.1,
            "sequences": 73656,
            "seqs_per_l": 256.6,
            "quality": 10,
        },
        "chosen": {
            "bitscore": 0.3,
            "sequences": 5816,
            "seqs_per_l": 15.4,
            "quality": 10,
        },
        "other_2": {
            "bitscore": 0.5,
            "sequences": 5566,
            "seqs_per_l": 14.8,
            "quality": 10,
        },
        "other_3": {
            "bitscore": 0.7,
            "sequences": 4476,
            "seqs_per_l": 11.8,
            "quality": 10,
        },
    },
}

def get_msa_dict():
    msa_dict = {}

    for l, v in EV_META.items():
        if l != "TrpB":
            if "chosen" in v.keys():
                msa_dict[l] = v["chosen"]["sequences"]
            else:
                msa_dict[l] = v["recommended"]["sequences"]
        else:
            ls = [
                l + "3" + app for app in [chr(i) for i in range(ord("A"), ord("I") + 1)]
            ] + ["TrpB4"]
            for t in ls:
                if "chosen" in v.keys():
                    msa_dict[t] = v["chosen"]["sequences"]
                else:
                    msa_dict[t] = v["recommended"]["sequences"]

    return msa_dict

MSA_DICT = deepcopy(get_msa_dict())


def get_msa_df() -> pd.DataFrame:
    """
    A function for getting the MSA dataframe for all landscapes
    """

    msa_df = pd.DataFrame(MSA_DICT, index=["MSA"]).T.reset_index()
    msa_df.columns = ["lib", "msa"]
    return msa_df

MSA_DF = get_msa_df()


def chop_pdb(
    input_pdb: str, output_pdb: str, start_resid: int, end_resid: int, chain_id: str
) -> None:
    """
    A function for chopping a pdb file to a specific chain and residue range

    Args:
    - input_pdb: str, path to the input pdb file
    - output_pdb: str, path to the output pdb file
    - start_resid: int, starting residue ID
    - end_resid: int, ending residue ID
    - chain_id: str, chain ID
    """

    # Initialize the parser and structure
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("structure", input_pdb)

    # Initialize the writer
    io = PDB.PDBIO()

    # Define a select class to filter the residues in the specific chain
    class ResidueSelect(PDB.Select):
        def accept_residue(self, residue):
            # Only accept residues in the specified chain with a residue ID greater than or equal to start_resid
            if (
                residue.parent.id == chain_id
                and residue.id[1] >= start_resid
                and residue.id[1] <= end_resid
            ):
                return True
            return False

    # Save the chopped structure to the output file
    io.set_structure(structure)
    io.save(output_pdb, ResidueSelect())

    print(
        f"Saved chopped structure starting from residue {start_resid} in chain {chain_id} to {output_pdb}"
    )


def pdb2seq(pdb_file_path: str, chain_id: str = "A") -> str:

    """
    A function for extracting chain in string format from pdb

    Args:
    - pdb_file_path: str,
    - chain_id: str = "A"
    """

    chains = {
        record.id: record.seq for record in SeqIO.parse(pdb_file_path, "pdb-atom")
    }

    return str(chains[[chain for chain in chains.keys() if chain_id in chain][0]])


def find_missing_str(longer: str, shorter: str) -> [str, str]:
    """
    A function for finding the missing part of a string

    Args:
    - longer: str, longer string
    - shorter: str, shorter string

    Returns:
    - part_before: str, part of the longer string before the shorter
    - part_after: str, part of the longer string after the shorter
    """
    # Find the start index of the shorter in the longer string
    start_index = longer.find(shorter)

    # If the shorter is not found, return the longer string as the "missing" part
    if start_index == -1:
        return "", ""

    # Find the end index of the shorter
    end_index = start_index + len(shorter)

    # Extract parts of the longer string that are not the shorter
    part_before = longer[:start_index]
    part_after = longer[end_index:]

    return part_before, part_after


def alignmutseq2pdbseq(mut_seq: str, pdb_seq: str) -> list[int]:
    """
    A function for aligning mutation sequence to pdb sequence and
    return the indices of the aligned sequence so that the mutation
    sequence can be trimmed to the lenght of the pdb sequence

    Args:
    - mut_seq: str, mutation sequence
    - pdb_seq: str, pdb sequence

    Returns:
    - list[int], start and end indices of the aligned sequence
    - pdb_seq: str, aligned pdb sequence
    """

    # Define a custom scoring function so that X is aligned with anything
    def custom_match_function(x, y):
        if x == "X" or y == "X":
            return 2  # High score for aligning X with anything
        elif x == y:
            return 2  # Match score
        else:
            return -1  # Mismatch score

    _, aligned_pdb_seq, _, _, _ = pairwise2.align.globalcs(
        mut_seq, pdb_seq, custom_match_function, -0.5, -0.1
    )[0]

    return [
        aligned_pdb_seq.find(aligned_pdb_seq.replace("-", "")[:1]),
        aligned_pdb_seq.rfind(aligned_pdb_seq.replace("-", "")[-1]),
    ], aligned_pdb_seq


def mut_csv2fasta(lib: str, ev_esm_dir: str = "ev_esm2") -> None:
    """
    A function for converting mutation csv to fasta

    Args:
    - lib: str, path to mutation csv
    - ev_esm_dir: str = "ev_esm2"
    """

    csv_path = f"{ev_esm_dir}/{lib}/{lib}.csv"

    if "TrpB" in lib:
        protein = "TrpB"
    else:
        protein = lib

    seq = SeqIO.read(f"data/{protein}/{protein}.fasta", "fasta").seq

    if lib == "DHFR":
        seq = str(seq.translate())
    else:
        seq = str(seq)

    pdb_path = f"data/{protein}/{protein}.pdb"
    processed_pdb_path = f"data/{protein}/{protein}_processed.pdb"

    if os.path.exists(processed_pdb_path):
        pdb_path = processed_pdb_path

    pdb_seq = pdb2seq(pdb_path, "A")

    df = pd.read_csv(csv_path)

    for col in ["muts", "seq"]:
        if col not in df.columns:
            raise ValueError(f"{col} column not found")

    fasta = csv_path.replace(".csv", ".fasta")

    print(f"Writing to {fasta}...")

    # pdb has more than fasta should only be for dhfr
    # TODO: find_missing_str and alignmutseq2pdbseq should be combined and improved
    if len(seq) < len(pdb_seq):
        print("PDB seq is longer than fasta")
        part_before, part_after = find_missing_str(longer=pdb_seq, shorter=seq)
        with open(fasta, "w") as f:
            for mut, seq in zip(df["muts"].values, df["seq"].values):
                f.write(f">{mut}\n{part_before+seq+part_after}\n")
    elif len(seq) == len(pdb_seq):
        print("PDB seq length is equal to fasta")
        with open(fasta, "w") as f:
            for mut, seq in zip(df["muts"].values, df["seq"].values):
                f.write(f">{mut}\n{seq}\n")
    else:
        print("Fasta seq is longer than PDB")
        index_list, aligned_pdb_seq = alignmutseq2pdbseq(mut_seq=seq, pdb_seq=pdb_seq)

        start_index, end_index = index_list

        # there might be seq with X from pdb
        # Step 1: Find all indices of 'X' in pdb_seq
        x_indices = [i for i, letter in enumerate(aligned_pdb_seq) if letter == "X"]

        with open(fasta, "w") as f:
            for mut, seq in zip(df["muts"].values, df["seq"].values):
                # Step 2: Modify the original seq by replacing characters at the found indices with 'X'
                if len(x_indices) > 0 and x_indices[-1] > start_index:
                    start_index = x_indices[-1] + 1
                    # seq_list = list(seq)  # Convert the sequence to a list to allow mutation
                    # for idx in x_indices:
                    #     seq_list[idx] = 'X'

                    # # Convert the modified list back to a string
                    # seq = ''.join(seq_list)
                f.write(f">{mut}\n{seq[start_index:end_index+1]}\n")


def get_all_mutfasta(
    ev_esm_dir: str = "ev_esm2", all_libs: bool = True, lib_list: list[str] = []
) -> None:
    """
    A function for converting all mutation csv to fasta
    subject to the pdb file sequence

    Args:
    - ev_esm_dir: str = "ev_esm2"
    - all_libs: bool = True
    - lib_list: list[str] = []
    """

    if all_libs:
        lib_list = LIB_INFO_DICT.keys()
    else:
        lib_list = deepcopy(lib_list)

    for lib in lib_list:
        print(f"Processing {lib}...")

        # if pdb_resrange is a key in the lib_info_dict
        if "pdb_resrange" in LIB_INFO_DICT[lib].keys():

            pdb_resrange = LIB_INFO_DICT[lib]["pdb_resrange"]
            chop_pdb(
                input_pdb=f"data/{lib}/{lib}.pdb",
                output_pdb=f"data/{lib}/{lib}_processed.pdb",
                start_resid=pdb_resrange[0],
                end_resid=pdb_resrange[1],
                chain_id="A",
            )

        mut_csv2fasta(lib, ev_esm_dir)


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
        return data

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

