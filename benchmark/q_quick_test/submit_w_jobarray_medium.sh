#!/bin/sh

#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3G
#SBATCH -n 1# Se solicita n tareas
#SBATCH -t 8:00:00 
#SBATCH -C clk
#SBATCH --array=3,8,13,18
# SBATCH --gres=gpu 

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
module load cesga/2020 gcc/system myqlm/1.7.3-python-3.9.9 pandas/1.3.5-python-3.9.9 gcccore/system plotly/5.6.0-python-3.9.9


srun -n 1 -c $SLURM_CPUS_PER_TASK python q_quick.py -id $SLURM_ARRAY_TASK_ID -repetitions 20 --print --save --exe 
