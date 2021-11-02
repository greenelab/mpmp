#!/bin/bash

#SBATCH -N 1
#SBATCH -t 0-12:00
#SBATCH --array=0-84%10
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_output/slurm-%A_%a.out
#SBATCH --error=slurm_output/slurm-%A_%a.err

# activate conda env
eval "$(conda shell.bash hook)"
conda activate ../mpmp-env
echo "Environment loaded"

# run command for a given gene in the list
# TODO: pull this from text file somewhere
genes=(
  "TP53"
  "TTN"
  "MUC16"
  "PIK3CA"
  "CSMD3"
  "RYR2"
  "LRP1B"
  "SYNE1"
  "FLG"
  "USH2A"
  "PCLO"
  "ZFHX4"
  "DNAH5"
  "KMT2D"
  "OBSCN"
  "CSMD1"
  "FAT4"
  "SPTA1"
  "KMT2C"
  "FAT3"
  "XIRP2"
  "HMCN1"
  "GPR98"
  "RYR1"
  "RYR3"
  "MUC17"
  "ARID1A"
  "NEB"
  "PCDH15"
  "PKHD1L1"
  "APC"
  "PTEN"
  "APOB"
  "LRP2"
  "DNAH9"
  "DNAH8"
  "SSPO"
  "KRAS"
  "ABCA13"
  "DMD"
  "BRAF"
  "AHNAK2"
  "DNAH11"
  "DNAH7"
  "MUC5B"
  "CSMD2"
  "FAT1"
  "XIST"
  "COL11A1"
  "SYNE2"
  "RELN"
  "PKHD1"
  "CUBN"
  "ANK2"
  "NAV3"
  "HYDIN"
  "DNAH3"
  "FAM135B"
  "AHNAK"
  "PAPPA2"
  "CACNA1E"
  "DST"
  "ANK3"
  "ATRX"
  "HERC2"
  "SI"
  "DNAH10"
  "TENM1"
  "BAGE2"
  "RP1"
  "HUWE1"
  "GPR112"
  "COL6A3"
  "LAMA2"
  "MACF1"
  "DNAH17"
  "KIAA1109"
  "MXRA5"
  "NF1"
  "FCGBP"
  "MDN1"
  "SACS"
  "FAT2"
  "UNC13C"
  "TPTE"
)

gene=${genes[${SLURM_ARRAY_TASK_ID}]}
results_dir=02_classify_mutations/results/expression_top

for seed in 42 1; do
    cmd="python 02_classify_mutations/run_mutation_classification.py "
    cmd+="--gene_set custom "
    cmd+="--custom_genes ${gene} "
    cmd+="--overlap_data_types expression "
    cmd+="--results_dir $results_dir "
    cmd+="--seed $seed "
    cmd+="--training_data expression "
    cmd+="--verbose"
    echo "Running: $cmd"
    eval $cmd
done

echo "Job complete"
