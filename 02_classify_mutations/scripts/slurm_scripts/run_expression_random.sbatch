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
  "PKHD1"
  "OTOGL"
  "TP63"
  "STAB2"
  "CDH6"
  "XIRP2"
  "BAP1"
  "DSG4"
  "ZDBF2"
  "NLRP5"
  "NRXN1"
  "DYSF"
  "LRP2"
  "PCSK5"
  "DOCK3"
  "KCNH5"
  "GNA11"
  "TSHZ3"
  "VCAN"
  "NSD1"
  "CACNA1A"
  "C12orf55"
  "AFF2"
  "CTNNB1"
  "ANK2"
  "CDH18"
  "FLG"
  "TPTE"
  "ABCA12"
  "SCN10A"
  "DNAH8"
  "ASPM"
  "SPHKAP"
  "COL5A2"
  "FBN2"
  "SPTA1"
  "KIAA2022"
  "CACNA2D3"
  "THSD7B"
  "SACS"
  "COL7A1"
  "DSG3"
  "DST"
  "GRIN2A"
  "FBXW7"
  "ARID1A"
  "BAGE2"
  "COL5A1"
  "CSMD3"
  "CDH9"
  "ATRX"
  "COL3A1"
  "SMAD4"
  "MACF1"
  "ERBB3"
  "RB1"
  "MUC16"
  "COL4A1"
  "ADAMTS12"
  "KCNT2"
  "CACNA1S"
  "OBSCN"
  "COL4A5"
  "PDE4DIP"
  "NDST4"
  "MYOCD"
  "TP53"
  "EPPK1"
  "ZNF536"
  "ADAM18"
  "ANO4"
  "FCRL5"
  "DNAH2"
  "IDH1"
  "CNTN4"
  "NEBL"
  "CPED1"
  "HUWE1"
  "SPAG17"
  "MYH7"
  "CTNND2"
  "NEB"
  "TCF7L2"
  "MYH2"
  "NBEAL2"
)

gene=${genes[${SLURM_ARRAY_TASK_ID}]}
results_dir=02_classify_mutations/results/expression_random

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
