#!/bin/sh

#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3G
#SBATCH -n 1# Se solicita n tareas
#SBATCH -t 3:00:00 
#SBATCH --array=0-27
#SBATCH --gres=gpu 

#Funciones
####################################################
CREATEFOLDER()	{
	Folder=$1
	if [ ! -d $Folder ]
	then
		mkdir $Folder
	fi
}
####################################################

echo SLURM_NTASKS: $SLURM_NTASKS  
echo SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK 
echo SLURM_NNODES: $SLURM_NNODES
echo SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE
echo 'SLURM_ARRAY_TASK_ID: '$SLURM_ARRAY_TASK_ID

#For saving DataFrames
FOLDER="/mnt/netapp1/Store_CESGA/home/cesga/gferro/Datos_2022_10_06/"
CREATEFOLDER $FOLDER
FILENAME="IQAE_2022_10_06_encoding1.csv"

module load myqlm

srun -n 1 -c $SLURM_CPUS_PER_TASK python ../benchmark_ae_estimation_price.py --IQAE --folder $FOLDER --name $FILENAME --save --exe -id $SLURM_ARRAY_TASK_ID 
