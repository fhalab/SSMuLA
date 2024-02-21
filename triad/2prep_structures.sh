TRIAD_DIR=/home/bwittmann/triad/triad-2.1.3
NAME=DHFR
ORIG_PDB_DIR=~/triad_struct/data/${NAME}

#add hydrogens
#${TRIAD_DIR}/triad.sh ${TRIAD_DIR}/apps/preparation/addH.py -struct ${ORIG_PDB_DIR} -crosetta -minimization False -optimizePolarH False -optimizeCarboxamides False -optimizeHistidineH False

cd ${ORIG_PDB_DIR}
# prepare initial structure
${TRIAD_DIR}/triad.sh ${TRIAD_DIR}/apps/preparation/proteinProcess.py -struct ${ORIG_PDB_DIR}/${NAME}.pdb -crosetta 

