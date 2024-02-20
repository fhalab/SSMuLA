# His tag in lowercase
TRANSLATE_DICT = {
    'AAA': 'K', 'AAT': 'N', 'AAC': 'N', 'AAG': 'K', 'ATA': 'I', 'ATT': 'I', 'ATC': 'I', 'ATG': 'M', 'ACA': 'T', 'ACT': 'T', 'ACC': 'T', 'ACG': 'T', 'AGA': 'R', 'AGT': 'S', 'AGC': 'S', 'AGG': 'R', 'TAA': '*', 'TAT': 'Y', 'TAC': 'Y', 'TAG': '*', 'TTA': 'L', 'TTT': 'F', 'TTC': 'F', 'TTG': 'L', 'TCA': 'S', 'TCT': 'S', 'TCC': 'S', 'TCG': 'S', 'TGA': '*', 'TGT': 'C', 'TGC': 'C', 'TGG': 'W', 'CAA': 'Q', 'CAT': 'H', 'CAC': 'H', 'CAG': 'Q', 'CTA': 'L', 'CTT': 'L', 'CTC': 'L', 'CTG': 'L', 'CCA': 'P', 'CCT': 'P', 'CCC': 'P', 'CCG': 'P', 'CGA': 'R', 'CGT': 'R', 'CGC': 'R', 'CGG': 'R', 'GAA': 'E', 'GAT': 'D', 'GAC': 'D', 'GAG': 'E', 'GTA': 'V', 'GTT': 'V', 'GTC': 'V', 'GTG': 'V', 'GCA': 'A', 'GCT': 'A', 'GCC': 'A', 'GCG': 'A', 'GGA': 'G', 'GGT': 'G', 'GGC': 'G', 'GGG': 'G'
}

TM9D8S_DNA = "ATGAAAGGCTACTTCGGTCCGTACGGTGGCCAGTACGTGCCGGAAATCCTGATGGGAGCTCTGGAAGAACTGGAAGCTGCGTACGAAGGAATCATGAAAGATGAGTCTTTCTGGAAAGAATTCAATGACCTGCTGCGCGATTATGCGGGTCGTCCGACTCCGCTGTACTTCGCACGTCGTCTGTCCGAAAAATACGGTGCTCGCGTATATCTGAAACGTGAAGACCTGCTGCATACTGGTGCGCATAAAATCAATAACGCTATCGGCCAGGTTCTGCTGGCAAAACTAATGGGCAAAACCCGTATCATTGCTGAAACGGGTGCTGGTCAGCACGGCGTAGCAACTGCTACCGCAGCAGCGCTGTTCGGTATGGAATGTGTAATCTATATGGGCGAAGAAGACACGATCCGCCAGAAACTAAACGTTGAACGTATGAAACTGCTGGGTGCTAAAGTTGTACCGGTAAAATCCGGTAGCCGTACCCTGAAAGACGCAATTGACGAAGCTCTGCGTGACTGGATTACCAACCTGCAGACCACCTATTACGTGTTCGGCTCTGTGGTTGGTCCGCATCCATATCCGATTATCGTACGTAACTTCCAAAAGGTTATCGGCGAAGAGACCAAAAAACAGATTCCAGAAAAAGAAGGCCGTCTGCCGGACTACATCGTTGCGTGCGTGAGCGGTGGTTCTAACGCTGCCGGTATCTTCTATCCGTTTATCGATTCTGGTGTGAAGCTGATCGGCGTAGAAGCCGGTGGCGAAGGTCTGGAAACCGGTAAACATGCGGCTTCTCTGCTGAAAGGTAAAATCGGCTACCTGCACGGTTCTAAGACGTTCGTTCTGCAGGATGACTGGGGTCAAGTTCAGGTGAGCCACTCCGTCTCCGCTGGCCTGGACTACTCCGGTGTCGGTCCGGAACACGCCTATTGGCGTGAGACCGGTAAAGTGCTGTACGATGCTGTGACCGATGAAGAAGCTCTGGACGCATTCATCGAACTGTCTCGCCTGGAAGGCATCATCCCAGCCCTGGAGTCTTCTCACGCACTGGCTTATCTGAAGAAGATCAACATCAAGGGTAAAGTTGTGGTGGTTAATCTGTCTGGTCGTGGTGACAAGGATCTGGAATCTGTACTGAACCACCCGTATGTTCGCGAACGCATCCGCctcgagcaccaccaccaccaccactga"

# His tag is LEHHHHHH*
TM9D8S_AA = "MKGYFGPYGGQYVPEILMGALEELEAAYEGIMKDESFWKEFNDLLRDYAGRPTPLYFARRLSEKYGARVYLKREDLLHTGAHKINNAIGQVLLAKLMGKTRIIAETGAGQHGVATATAAALFGMECVIYMGEEDTIRQKLNVERMKLLGAKVVPVKSGSRTLKDAIDEALRDWITNLQTTYYVFGSVVGPHPYPIIVRNFQKVIGEETKKQIPEKEGRLPDYIVACVSGGSNAAGIFYPFIDSGVKLIGVEAGGEGLETGKHAASLLKGKIGYLHGSKTFVLQDDWGQVQVSHSVSAGLDYSGVGPEHAYWRETGKVLYDAVTDEEALDAFIELSRLEGIIPALESSHALAYLKKINIKGKVVVVNLSGRGDKDLESVLNHPYVRERIRLEHHHHHH*"

# Dictionary with positions for mutation, parent codons, and parent amino acids
LIBRARY_INFO_DICT = {
    'A': {
        'positions': {1: 104, 2: 105, 3: 106},
        'codons': {1: 'GCT', 2: 'GAA', 3: 'ACG'}, 
        'AAs': {1: 'A', 2: 'E', 3: 'T'}
        },
    'B': {
        'positions': {1: 105, 2: 106, 3: 107},
        'codons': {1: 'GAA', 2: 'ACG', 3: 'GGT'}, 
        'AAs': {1: 'E', 2: 'T', 3: 'G'}
    },
    'C': {
        'positions': {1: 106, 2: 107, 3: 108},
        'codons': {1: 'ACG', 2: 'GGT', 3: 'GCT'}, 
        'AAs': {1: 'T', 2: 'G', 3: 'A'}
    },
    'D': {
        'positions': {1: 117, 2: 118, 3: 119},
        'codons': {1: 'ACC', 2: 'GCA', 3: 'GCA'}, 
        'AAs': {1: 'T', 2: 'A', 3: 'A'}
        },
    'E': {
        'positions': {1: 184, 2: 185, 3: 186},
        'codons': {1: 'TTC', 2: 'GGC', 3: 'TCT'}, 
        'AAs': {1: 'F', 2: 'G', 3: 'S'}
    },
    'F': {
        'positions': {1: 162, 2: 166, 3: 301},
        'codons': {1: 'CTG', 2: 'ATT', 3: 'TAC'}, 
        'AAs': {1: 'L', 2: 'I', 3: 'Y'}
    },
    'G': {
        'positions': {1: 227, 2: 228, 3: 301},
        'codons': {1: 'GTG', 2: 'AGC', 3: 'TAC'}, 
        'AAs': {1: 'V', 2: 'S', 3: 'Y'}
    },
    'H': {
        'positions': {1: 228, 2: 230, 3: 231},
        'codons': {1: 'AGC', 2: 'GGT', 3: 'TCT'}, 
        'AAs': {1: 'S', 2: 'G', 3: 'S'}
    },
    'I': {
        'positions': {1: 182, 2: 183, 3: 184},
        'codons': {1: 'TAC', 2: 'GTG', 3: 'TTC'}, 
        'AAs': {1: 'Y', 2: 'V', 3: 'F'}
    },
    '4-site': {
        'positions': {1: 183, 2: 184, 3: 227, 4: 228},
        'codons': {1: 'GTG', 2: 'TTC', 3: 'GTG', 4: 'AGC'},
        'AAs': {1: 'V', 2: 'F', 3: 'V', 4: 'S'}
    }
}

# Dictionary encoding the ODs over time for libraries DEFGHI
OD_DICT = {
    'A': {
        'rep_1': {
            0: 0.1,
            18: 0.72,
            20: 0.78,
            24: 0.94,
            44: 2.55
        },
        'rep_2': {
            0: 0.1,
            18: 0.75,
            20: 0.83,
            24: 1.01,
            44: 2.70
        },
    },
    'B': {
        'rep_1': {
            0: 0.1,
            18: 0.75,
            20: 0.83,
            24: 1.09,
            44: 3.30
        },
        'rep_2': {
            0: 0.1,
            18: 0.84,
            20: 0.98,
            24: 1.50,
            44: 3.85
        },
    },
    'C': {
        'rep_1': {
            0: 0.1,
            18: 0.74,
            20: 0.78,
            24: 0.86,
            44: 1.95
        },
        'rep_2': {
            0: 0.1,
            18: 0.76,
            20: 0.84,
            24: 0.92,
            44: 4.15
        },
    },
    'D': {
        'rep_1': {
            0: 0.05,
            12: 0.19,
            16: 0.29,
            20: 0.51,
            24: 0.85,
            36: 1.42
        },
        'rep_2': {
            0: 0.05,
            12: 0.18,
            16: 0.28,
            20: 0.49,
            24: 0.97,
            36: 1.81
        }
    },
    'E': {
        'rep_1': {
            0: 0.05,
            12: 0.2,
            16: 0.27,
            20: 0.47,
            24: 0.91,
            36: 1.41
        },
        'rep_2': {
            0: 0.05,
            12: 0.2,
            16: 0.26,
            20: 0.44,
            24: 0.94,
            36: 1.54
        }
    },
    'F': {
        'rep_1': {
            0: 0.05,
            12: 0.17,
            16: 0.20,
            20: 0.23,
            24: 0.27,
            36: 0.79
        },
        'rep_2': {
            0: 0.05,
            12: 0.17,
            16: 0.20,
            20: 0.24,
            24: 0.27,
            36: 0.79
        }
    },
    'G': {
        'rep_1': {
            0: 0.05,
            12: 0.14,
            16: 0.18,
            20: 0.23,
            24: 0.44,
            36: 1.95
        },
        'rep_2': {
            0: 0.05,
            12: 0.14,
            16: 0.18,
            20: 0.23,
            24: 0.44,
            36: 1.95
        }
    },
    'H': {
        'rep_1': {
            0: 0.05,
            12: 0.15,
            16: 0.19,
            20: 0.26,
            24: 0.67,
            36: 2.90
        },
        'rep_2': {
            0: 0.05,
            12: 0.14,
            16: 0.18,
            20: 0.26,
            24: 0.58,
            36: 1.85
        }
    },
    'I': {
        'rep_1': {
            0: 0.05,
            12: 0.36,
            16: 0.83,
            20: 1.24,
            24: 0.7,
            36: 1.95
        },
        'rep_2': {
            0: 0.05,
            12: 0.39,
            16: 0.87,
            20: 1.36,
            24: 2.1,
            36: 2.25
        }
    },
    '4-site': {
        'rep_1': {
            0: 0.025,
            12: 0.19,
            16: 0.51,
            20: 1.26,
            24: 1.50,
            28: 1.675,
            36: 1.75
        },
        'rep_2': {
            0: 0.025,
            12: 0.19,
            16: 0.52,
            20: 1.34,
            24: 1.625,
            28: 1.75,
            36: 1.875

        }
    },
}

TIMEPOINT_DICT = {
    'A': {
        'T0': 0,
        'T1': 18,
        'T2': 20,
        'T3': 24,
        'T4': 44
    },
    'B': {
        'T0': 0,
        'T1': 18,
        'T2': 20,
        'T3': 24,
        'T4': 44
    },
    'C': {
        'T0': 0,
        'T1': 18,
        'T2': 20,
        'T3': 24,
        'T4': 44
    },
    'D': {
        'T0': 0,
        'T1': 12,
        'T2': 16,
        'T3': 20,
        'T4': 24,
        'T5': 36
    },
    'E': {
        'T0': 0,
        'T1': 12,
        'T2': 16,
        'T3': 20,
        'T4': 24,
        'T5': 36
    },
    'F': {
        'T0': 0,
        'T1': 12,
        'T2': 16,
        'T3': 20,
        'T4': 24,
        'T5': 36
    },
    'G': {
        'T0': 0,
        'T1': 12,
        'T2': 16,
        'T3': 20,
        'T4': 24,
        'T5': 36
    },
    'H': {
        'T0': 0,
        'T1': 12,
        'T2': 16,
        'T3': 20,
        'T4': 24,
        'T5': 36
    },
    'I': {
        'T0': 0,
        'T1': 12,
        'T2': 16,
        'T3': 20,
        'T4': 24,
        'T5': 36
    },
    '4-site': {
        'T0': 0,
        'T1': 12,
        'T2': 16,
        'T3': 20,
        'T4': 24,
        'T5': 28,
        'T6': 36
    }
}

# THREESITE_OD_DICT = {
#     'A': {
#         'rep_1': {
#             0: 0.1,
#             18: 0.72,
#             20: 0.78,
#             24: 0.94,
#             44: 2.55
#         },
#         'rep_2': {
#             0: 0.1,
#             18: 0.75,
#             20: 0.83,
#             24: 1.01,
#             44: 2.70
#         },
#     },
#     'B': {
#         'rep_1': {
#             0: 0.1,
#             18: 0.75,
#             20: 0.83,
#             24: 1.09,
#             44: 3.30
#         },
#         'rep_2': {
#             0: 0.1,
#             18: 0.84,
#             20: 0.98,
#             24: 1.50,
#             44: 3.85
#         },
#     },
#     'C': {
#         'rep_1': {
#             0: 0.1,
#             18: 0.74,
#             20: 0.78,
#             24: 0.86,
#             44: 1.95
#         },
#         'rep_2': {
#             0: 0.1,
#             18: 0.76,
#             20: 0.84,
#             24: 0.92,
#             44: 4.15
#         },
#     },
#     'D': {
#         'rep_1': {
#             0: 0.05,
#             12: 0.19,
#             16: 0.29,
#             20: 0.51,
#             24: 0.85,
#             36: 1.42
#         },
#         'rep_2': {
#             0: 0.05,
#             12: 0.18,
#             16: 0.28,
#             20: 0.49,
#             24: 0.97,
#             36: 1.81
#         }
#     },
#     'E': {
#         'rep_1': {
#             0: 0.05,
#             12: 0.2,
#             16: 0.27,
#             20: 0.47,
#             24: 0.91,
#             36: 1.41
#         },
#         'rep_2': {
#             0: 0.05,
#             12: 0.2,
#             16: 0.26,
#             20: 0.44,
#             24: 0.94,
#             36: 1.54
#         }
#     },
#     'F': {
#         'rep_1': {
#             0: 0.05,
#             12: 0.17,
#             16: 0.20,
#             20: 0.23,
#             24: 0.27,
#             36: 0.79
#         },
#         'rep_2': {
#             0: 0.05,
#             12: 0.17,
#             16: 0.20,
#             20: 0.24,
#             24: 0.27,
#             36: 0.79
#         }
#     },
#     'G': {
#         'rep_1': {
#             0: 0.05,
#             12: 0.14,
#             16: 0.18,
#             20: 0.23,
#             24: 0.44,
#             36: 1.95
#         },
#         'rep_2': {
#             0: 0.05,
#             12: 0.14,
#             16: 0.18,
#             20: 0.23,
#             24: 0.44,
#             36: 1.95
#         }
#     },
#     'H': {
#         'rep_1': {
#             0: 0.05,
#             12: 0.15,
#             16: 0.19,
#             20: 0.26,
#             24: 0.67,
#             36: 2.90
#         },
#         'rep_2': {
#             0: 0.05,
#             12: 0.14,
#             16: 0.18,
#             20: 0.26,
#             24: 0.58,
#             36: 1.85
#         }
#     },
#     'I': {
#         'rep_1': {
#             0: 0.05,
#             12: 0.36,
#             16: 0.83,
#             20: 1.24,
#             24: 0.7,
#             36: 1.95
#         },
#         'rep_2': {
#             0: 0.05,
#             12: 0.39,
#             16: 0.87,
#             20: 1.36,
#             24: 2.1,
#             36: 2.25
#         }
#     }
# }

# foursite_OD_dict = {
#     '4-site': {
#         'rep_1': {
#             0: 0.025,
#             12: 0.19,
#             16: 0.51,
#             20: 1.26,
#             24: 1.50,
#             28: 1.675,
#             36: 1.75
#         },
#         'rep_2': {
#             0: 0.025,
#             12: 0.19,
#             16: 0.52,
#             20: 1.34,
#             24: 1.625,
#             28: 1.75,
#             36: 1.875

#         }
#     },
# }

REFERENCE_READS = {
    'A': {
        'fwd': 'NNNNNNNNNNXXX'+ 'GCCAGGTTCTGCTGGCAAAACTAATGGGCAAAACCCGTATCATTGCTGAAACGGGTGCTGGTCAGCACGGCGTAGCAACTGCTACCGCAGCAGCGCTGTTCGGTATGGAATGTGTAATCTATATGGGCGAAGAAGAC',
        'rev': 'NNNNNNNNNNXXX'+ 'TCTGGCGGATCGTGTCTTCTTCGCCCATATAGATTACACATTCCATACCGAACAGCGCTGCTGCGGTAGCAGTTGCTACGCCGTGCTGACCAGCACCCGTTTCAGCAATGATACGGGTTTTGCCCATTAGTTTTGCC'
    },
    'B': {
        'fwd': 'NNNNNNNNNNXXX'+ 'GCCAGGTTCTGCTGGCAAAACTAATGGGCAAAACCCGTATCATTGCTGAAACGGGTGCTGGTCAGCACGGCGTAGCAACTGCTACCGCAGCAGCGCTGTTCGGTATGGAATGTGTAATCTATATGGGCGAAGAAGAC',
        'rev': 'NNNNNNNNNNXXX'+ 'TCTGGCGGATCGTGTCTTCTTCGCCCATATAGATTACACATTCCATACCGAACAGCGCTGCTGCGGTAGCAGTTGCTACGCCGTGCTGACCAGCACCCGTTTCAGCAATGATACGGGTTTTGCCCATTAGTTTTGCC'
    },
    'C': {
        'fwd': 'NNNNNNNNNNXXX'+ 'GCCAGGTTCTGCTGGCAAAACTAATGGGCAAAACCCGTATCATTGCTGAAACGGGTGCTGGTCAGCACGGCGTAGCAACTGCTACCGCAGCAGCGCTGTTCGGTATGGAATGTGTAATCTATATGGGCGAAGAAGAC',
        'rev': 'NNNNNNNNNNXXX'+ 'TCTGGCGGATCGTGTCTTCTTCGCCCATATAGATTACACATTCCATACCGAACAGCGCTGCTGCGGTAGCAGTTGCTACGCCGTGCTGACCAGCACCCGTTTCAGCAATGATACGGGTTTTGCCCATTAGTTTTGCC'
    },
    'D': {
        'fwd': 'NNNNNNNNNNXXX'+ 'GCCAGGTTCTGCTGGCAAAACTAATGGGCAAAACCCGTATCATTGCTGAAACGGGTGCTGGTCAGCACGGCGTAGCAACTGCTACCGCAGCAGCGCTGTTCGGTATGGAATGTGTAATCTATATGGGCGAAGAAGAC',
        'rev': 'NNNNNNNNNNXXX'+ 'TCTGGCGGATCGTGTCTTCTTCGCCCATATAGATTACACATTCCATACCGAACAGCGCTGCTGCGGTAGCAGTTGCTACGCCGTGCTGACCAGCACCCGTTTCAGCAATGATACGGGTTTTGCCCATTAGTTTTGCC'
    },
    'E': {
        'fwd': 'NNNNNNNNNNXXX'+ 'GCTAAAGTTGTACCGGTAAAATCCGGTAGCCGTACCCTGAAAGACGCAATTGACGAAGCTCTGCGTGACTGGATTACCAACCTGCAGACCACCTATTACGTGTTCGGCTCTGTGGTTGGTCCGCATCCATATCCGAT',
        'rev': 'NNNNNNNNNNXXX'+ 'CGATAATCGGATATGGATGCGGACCAACCACAGAGCCGAACACGTAATAGGTGGTCTGCAGGTTGGTAATCCAGTCACGCAGAGCTTCGTCAATTGCGTCTTTCAGGGTACGGCTACCGGATTTTACCGGTACAAC'
    },
    'F': {
        'fwd': 'NNNNNNNNNNXXX'+ 'GCTAAAGTTGTACCGGTAAAATCCGGTAGCCGTACCCTGAAAGACGCAATTGACGAAGCTCTGCGTGACTGGATTACCAACCTGCAGACCACCTATTACGTGTTCGGCTCTGTGGTTGGTCCGCATCCATATCCGAT',
        'rev': 'NNNNNNNNNNXXX'+ 'TAGGCGTGTTCCGGACCGACACCGGAGTAGTCCAGGCCAGCGGAGACGGAGTGGCTCACCTGAACTTGACCCCAGTCATCCTGCAGAACGAACGTCTTAGAACCGTGCAGGTAGCCGATTTTACCTTTCAGCAGAGA'
    },
    'G': {
        'fwd': 'NNNNNNNNNNXXX' + 'CGGACTACATCGTTGCGTGCGTGAGCGGTGGTTCTAAC',
        'rev': 'NNNNNNNNNNXXX' + 'TAGGCGTGTTCCGGACCGACACCGGAGTAGTCCAGGCC'
    },
    'H': {
        'fwd': 'NNNNNNNNNNXXX' + 'CGGACTACATCGTTGCGTGCGTGAGCGGTGGTTCTAAC',
        'rev': 'NNNNNNNNNNXXX' + 'TAGGCGTGTTCCGGACCGACACCGGAGTAGTCCAGGCC'
    },
    'I': {
        'fwd': 'NNNNNNNNNNXXX' + 'GCTAAAGTTGTACCGGTAAAATCCGGTAGCCGTACCCTGAAAGACGCAATTGACGAAGCTCTGCGTGACTGGATTACCAACCTGCAGACCACCTATTACGTGTTCGGCTCTGTGGTTGGTCCGCATCCATATCCGAT',
        'rev': 'NNNNNNNNNNXXX'+ 'CGATAATCGGATATGGATGCGGACCAACCACAGAGCCGAACACGTAATAGGTGGTCTGCAGGTTGGTAATCCAGTCACGCAGAGCTTCGTCAATTGCGTCTTTCAGGGTACGGCTACCGGATTTTACCGGTACAAC'
    },
    '4-site': {
        'fwd': 'NNNNNNNNNNXXX' + 'GGATTACCAACCTGCAGACCACCTATTACGTGTTCGGC',
        'rev': 'NNNNNNNNNNXXX' + 'AGATACCGGCAGCGTTAGAACCACCGCTCACGCACGCA'
    }
}

CODON_STARTS = {
    'A': {
        'fwd': {
            'codon1': 58,
            'codon2': 61,
            'codon3': 64
        },
        'rev': {
            
        }
    },
    'B': {
        'fwd': {
            'codon1': 61,
            'codon2': 64,
            'codon3': 67
        },
        'rev': {
            
        }
    },
    'C': {
        'fwd': {
            'codon1': 64,
            'codon2': 67,
            'codon3': 70
        },
        'rev': {
            
        }
    },
    'D': {
        'fwd': {
        },
        'rev': {
            'codon1': 78,
            'codon2': 75,
            'codon3': 72
        }
    },
    'E': {
        'fwd': {
        },
        'rev': {
            'codon1': 51,
            'codon2': 48,
            'codon3': 45
        }
    },
    'F': {
        'fwd': {
            'codon1': 50,
            'codon2': 62,
        },
        'rev': {
            'codon3': 40
        }
    },
    'G': {
        'fwd': {
            'codon1': 34,
            'codon2': 37,
        },
        'rev': {
            'codon3': 40
        }
    },
    'H': {
        'fwd': {
            'codon1': 37,
            'codon2': 43,
            'codon3': 46
        },
        'rev': {
            
        }
    },
    'I': {
        'fwd': {
        },
        'rev': {
            'codon1': 57,
            'codon2': 54,
            'codon3': 51
        }
    },
    '4-site': {
        'fwd': {
            'codon1': 43,
            'codon2': 46,
        },
        'rev': {
            'codon3': 42,
            'codon4': 39,
        }
    }
}

FASTQ_MAPPING = {
    '4-site': {
        '20230501': {
            'T0_rep1': {
                'fwd_file': '25561_*_S19_R1_001.fastq.gz',
                'rev_file': '25561_*_S19_R2_001.fastq.gz'
            },
            'T1_rep1': {
                'fwd_file': '25561_*_S20_R1_001.fastq.gz',
                'rev_file': '25561_*_S20_R2_001.fastq.gz'
            },
            'T2_rep1': {
                'fwd_file': '25561_*_S21_R1_001.fastq.gz',
                'rev_file': '25561_*_S21_R2_001.fastq.gz'
            },
            'T3_rep1': {
                'fwd_file': '25561_*_S22_R1_001.fastq.gz',
                'rev_file': '25561_*_S22_R2_001.fastq.gz'
            },
            'T4_rep1': {
                'fwd_file': '25561_*_S23_R1_001.fastq.gz',
                'rev_file': '25561_*_S23_R2_001.fastq.gz'
            },
            'T5_rep1': {
                'fwd_file': '25561_*_S24_R1_001.fastq.gz',
                'rev_file': '25561_*_S24_R2_001.fastq.gz'
            },
            'T6_rep1': {
                'fwd_file': '25561_*_S25_R1_001.fastq.gz',
                'rev_file': '25561_*_S25_R2_001.fastq.gz'
            },
            'T0_rep2': {
                'fwd_file': '25561_*_S26_R1_001.fastq.gz',
                'rev_file': '25561_*_S26_R2_001.fastq.gz'
            },
            'T1_rep2': {
                'fwd_file': '25561_*_S27_R1_001.fastq.gz',
                'rev_file': '25561_*_S27_R2_001.fastq.gz'
            },
            'T2_rep2': {
                'fwd_file': '25561_*_S28_R1_001.fastq.gz',
                'rev_file': '25561_*_S28_R2_001.fastq.gz'
            },
            'T3_rep2': {
                'fwd_file': '25561_*_S29_R1_001.fastq.gz',
                'rev_file': '25561_*_S29_R2_001.fastq.gz'
            },
            'T4_rep2': {
                'fwd_file': '25561_*_S30_R1_001.fastq.gz',
                'rev_file': '25561_*_S30_R2_001.fastq.gz'
            },
            'T5_rep2': {
                'fwd_file': '25561_*_S31_R1_001.fastq.gz',
                'rev_file': '25561_*_S31_R2_001.fastq.gz'
            },
            'T6_rep2': {
                'fwd_file': '25561_*_S32_R1_001.fastq.gz',
                'rev_file': '25561_*_S32_R2_001.fastq.gz'
            }},
        '20230523': {
            'T0_rep1': {
                'fwd_file': '25561_*_S1_R1_001.fastq.gz',
                'rev_file': '25561_*_S1_R2_001.fastq.gz'
            },
            'T1_rep1': {
                'fwd_file': '25561_*_S2_R1_001.fastq.gz',
                'rev_file': '25561_*_S2_R2_001.fastq.gz'
            },
            'T2_rep1': {
                'fwd_file': '25561_*_S3_R1_001.fastq.gz',
                'rev_file': '25561_*_S3_R2_001.fastq.gz'
            },
            'T3_rep1': {
                'fwd_file': '25561_*_S4_R1_001.fastq.gz',
                'rev_file': '25561_*_S4_R2_001.fastq.gz'
            },
            'T4_rep1': {
                'fwd_file': '25561_*_S5_R1_001.fastq.gz',
                'rev_file': '25561_*_S5_R2_001.fastq.gz'
            },
            'T5_rep1': {
                'fwd_file': '25561_*_S6_R1_001.fastq.gz',
                'rev_file': '25561_*_S6_R2_001.fastq.gz'
            },
            'T6_rep1': {
                'fwd_file': '25561_*_S7_R1_001.fastq.gz',
                'rev_file': '25561_*_S7_R2_001.fastq.gz'
            },
            'T0_rep2': {
                'fwd_file': '25561_*_S8_R1_001.fastq.gz',
                'rev_file': '25561_*_S8_R2_001.fastq.gz'
            },
            'T1_rep2': {
                'fwd_file': '25561_*_S9_R1_001.fastq.gz',
                'rev_file': '25561_*_S9_R2_001.fastq.gz'
            },
            'T2_rep2': {
                'fwd_file': '25561_*_S10_R1_001.fastq.gz',
                'rev_file': '25561_*_S10_R2_001.fastq.gz'
            },
            'T3_rep2': {
                'fwd_file': '25561_*_S11_R1_001.fastq.gz',
                'rev_file': '25561_*_S11_R2_001.fastq.gz'
            },
            'T4_rep2': {
                'fwd_file': '25561_*_S12_R1_001.fastq.gz',
                'rev_file': '25561_*_S12_R2_001.fastq.gz'
            },
            'T5_rep2': {
                'fwd_file': '25561_*_S13_R1_001.fastq.gz',
                'rev_file': '25561_*_S13_R2_001.fastq.gz'
            },
            'T6_rep2': {
                'fwd_file': '25561_*_S14_R1_001.fastq.gz',
                'rev_file': '25561_*_S14_R2_001.fastq.gz'
            }
        },
    }
}

SAMFILE_MAPPING = {
    '4-site':{
        '20230523': {
            'T0_rep1': 'trimmed_25561_*_S1.sam.gz',     
            'T1_rep1': 'trimmed_25561_*_S2.sam.gz',
            'T2_rep1': 'trimmed_25561_*_S3.sam.gz',
            'T3_rep1': 'trimmed_25561_*_S4.sam.gz',
            'T4_rep1': 'trimmed_25561_*_S5.sam.gz',
            'T5_rep1': 'trimmed_25561_*_S6.sam.gz',
            'T6_rep1': 'trimmed_25561_*_S7.sam.gz',
            'T0_rep2': 'trimmed_25561_*_S8.sam.gz',
            'T1_rep2': 'trimmed_25561_*_S9.sam.gz',
            'T2_rep2': 'trimmed_25561_*_S10.sam.gz',
            'T3_rep2': 'trimmed_25561_*_S11.sam.gz',
            'T4_rep2': 'trimmed_25561_*_S12.sam.gz',
            'T5_rep2': 'trimmed_25561_*_S13.sam.gz',
            'T6_rep2': 'trimmed_25561_*_S14.sam.gz',
        }
    },
    'A': {
        '20221102': {
            'T0_rep1': '*S6.sam.gz',
            'T1_rep1': '*S7.sam.gz',
            'T4_rep1': '*S8.sam.gz'
        }
    },
    'B': {
        '20221102': {
            'T0_rep1': '*S9.sam.gz',
            'T1_rep1': '*S10.sam.gz',
            'T4_rep1': '*S11.sam.gz'
        }
    },
    'C': {
        '20221102': {
            'T0_rep1': '*S12.sam.gz',
            'T1_rep1': '*S13.sam.gz',
            'T4_rep1': '*S14.sam.gz'
        }
    },
    'D': {
        '20221102': {
            'T0_rep1': '*S15.sam.gz',
            'T1_rep1': '*S16.sam.gz',
            'T2_rep1': '*S17.sam.gz',
            'T3_rep1': '*S18.sam.gz',
            'T4_rep1': '*S19.sam.gz',
            'T5_rep1': '*S20.sam.gz',
            'T1_rep2': '*S21.sam.gz',
            'T2_rep2': '*S22.sam.gz',
            'T3_rep2': '*S23.sam.gz',
            'T4_rep2': '*S24.sam.gz',
            'T5_rep2': '*S25.sam.gz'
        }
    },
    'E': {
        '20221102': {
            'T0_rep1': '*S26.sam.gz',
            'T1_rep1': '*S27.sam.gz',
            'T2_rep1': '*S28.sam.gz',
            'T3_rep1': '*S29.sam.gz',
            'T4_rep1': '*S30.sam.gz',
            'T5_rep1': '*S31.sam.gz',
            'T1_rep2': '*S32.sam.gz',
            'T2_rep2': '*S33.sam.gz',
            'T3_rep2': '*S34.sam.gz',
            'T4_rep2': '*S35.sam.gz',
            'T5_rep2': '*S36.sam.gz'
        }
    },
    'F': {
        '20221102': {
            'T0_rep1': '*S37.sam.gz',
            'T1_rep1': '*S38.sam.gz',
            'T2_rep1': '*S39.sam.gz',
            'T3_rep1': '*S40.sam.gz',
            'T4_rep1': '*S41.sam.gz',
            'T5_rep1': '*S42.sam.gz',
            'T1_rep2': '*S43.sam.gz',
            'T2_rep2': '*S44.sam.gz',
            'T3_rep2': '*S45.sam.gz',
            'T4_rep2': '*S46.sam.gz',
            'T5_rep2': '*S47.sam.gz'
        }
    },
    'G': {
        '20221102': {
            'T0_rep1': '*S48.sam.gz',
            'T1_rep1': '*S49.sam.gz',
            'T2_rep1': '*S50.sam.gz',
            'T3_rep1': '*S51.sam.gz',
            'T4_rep1': '*S52.sam.gz',
            'T5_rep1': '*S53.sam.gz',
            'T1_rep2': '*S54.sam.gz',
            'T2_rep2': '*S55.sam.gz',
            'T3_rep2': '*S56.sam.gz',
            'T4_rep2': '*S57.sam.gz',
            'T5_rep2': '*S58.sam.gz'
        }
    },
    'H': {
        '20221102': {
            'T0_rep1': '*S59.sam.gz',
            'T1_rep1': '*S60.sam.gz',
            'T2_rep1': '*S61.sam.gz',
            'T3_rep1': '*S62.sam.gz',
            'T4_rep1': '*S63.sam.gz',
            'T5_rep1': '*S64.sam.gz',
            'T1_rep2': '*S65.sam.gz',
            'T2_rep2': '*S66.sam.gz',
            'T3_rep2': '*S67.sam.gz',
            'T4_rep2': '*S68.sam.gz',
            'T5_rep2': '*S69.sam.gz'
        }
    },
    'I': {
        '20221102': {
            'T0_rep1': '*S70.sam.gz',
            'T1_rep1': '*S71.sam.gz',
            'T2_rep1': '*S72.sam.gz',
            'T3_rep1': '*S73.sam.gz',
            'T4_rep1': '*S74.sam.gz',
            'T5_rep1': '*S75.sam.gz',
            'T1_rep2': '*S76.sam.gz',
            'T2_rep2': '*S77.sam.gz',
            'T3_rep2': '*S78.sam.gz',
            'T4_rep2': '*S79.sam.gz',
            'T5_rep2': '*S80.sam.gz'
        }
    }
}

SAMFILE_CODONS = {
    '4-site': {
        'fwd': {
            'codon1': 29,
            'codon2': 32,
        },
        'rev': {
            'codon3': 7,
            'codon4': 10,
        }
    },
    'A': {
        'fwd': {
            'codon1': 58-14,
            'codon2': 61-14,
            'codon3': 64-14
        },
        'rev': {
            
        }
    },
    'B': {
        'fwd': {
            'codon1': 61-14,
            'codon2': 64-14,
            'codon3': 67-14
        },
        'rev': {
            
        }
    },
    'C': {
        'fwd': {
            'codon1': 64-14,
            'codon2': 67-14,
            'codon3': 70-14
        },
        'rev': {
            
        }
    },
    'D': {
        'fwd': {
        },
        # 'rev': {
        #     'codon1': 78,
        #     'codon2': 75,
        #     'codon3': 72
        # }
        'rev': {
            'codon1': 99-78,
            'codon2': 99-75,
            'codon3': 99-72
        }
    },
    'E': {
        'fwd': {
        },
        'rev': {
            'codon1': 99-51,
            'codon2': 99-48,
            'codon3': 99-45
        }
    },
    'F': {
        'fwd': {
            'codon1': 50-14,
            'codon2': 62-14,
        },
        'rev': {
            'codon3': 99-40
        }
    },
    'G': {
        'fwd': {
            'codon1': 34-14,
            'codon2': 37-14,
        },
        'rev': {
            'codon3': 99-40
        }
    },
    'H': {
        'fwd': {
            'codon1': 37-14,
            'codon2': 43-14,
            'codon3': 46-14
        },
        'rev': {
            
        }
    },
    'I': {
        'fwd': {
        },
        'rev': {
            'codon1': 99-57,
            'codon2': 99-54,
            'codon3': 99-51
        }
    },
}

PLOT_STYLE = {
    'attrs' : {
        'Title': {
            'align':'center',
            'text_font_size':'15px',
            'text_color':'black',
            'text_font': 'arial'
        },       # title centered and bigger
        'Axis': {
            'axis_label_text_font_style': 'normal',
            'axis_label_text_color':'black',
            'major_label_text_color':'black',
            'axis_label_text_font': 'arial',
            'major_label_text_font': 'arial',
        },          # no italic labels 
        'Legend': {
            'title_text_font_style': 'normal',
            'title_text_color':'black',
            'label_text_color':'black',
            'label_text_font': 'arial',
        },
        'ColorBar': {
            'title_text_font_style': 'normal',
            'major_label_text_color':'black',
            'major_label_text_font': 'arial',
            'title_text_color':'black',
            'title_text_font': 'arial',
        },
    }
}