CUDA_VISIBLE_DEVICES=1

python score_log_likelihoods.py DHFR.pdb DHFR.fasta --chain A --outpath DHFR_esmif_scores.csv
python score_log_likelihoods.py GB1.pdb GB1.fasta --chain A --outpath GB1_esmif_scores.csv
python score_log_likelihoods.py ParD2.pdb ParD2.fasta --chain A --outpath ParD2_esmif_scores.csv
python score_log_likelihoods.py ParD3.pdb ParD3.fasta --chain A --outpath ParD3_esmif_scores.csv
python score_log_likelihoods.py TrpB.pdb TrpB3A.fasta --chain A --outpath TrpB3A_esmif_scores.csv
python score_log_likelihoods.py TrpB.pdb TrpB3B.fasta --chain A --outpath TrpB3B_esmif_scores.csv
python score_log_likelihoods.py TrpB.pdb TrpB3C.fasta --chain A --outpath TrpB3C_esmif_scores.csv
python score_log_likelihoods.py TrpB.pdb TrpB3D.fasta --chain A --outpath TrpB3D_esmif_scores.csv
python score_log_likelihoods.py TrpB.pdb TrpB3E.fasta --chain A --outpath TrpB3E_esmif_scores.csv
python score_log_likelihoods.py TrpB.pdb TrpB3F.fasta --chain A --outpath TrpB3F_esmif_scores.csv
python score_log_likelihoods.py TrpB.pdb TrpB3G.fasta --chain A --outpath TrpB3G_esmif_scores.csv
python score_log_likelihoods.py TrpB.pdb TrpB3H.fasta --chain A --outpath TrpB3H_esmif_scores.csv
python score_log_likelihoods.py TrpB.pdb TrpB3I.fasta --chain A --outpath TrpB3I_esmif_scores.csv
python score_log_likelihoods.py TrpB.pdb TrpB4.fasta --chain A --outpath TrpB4_esmif_scores.csv