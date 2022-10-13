#!/bin/sh


#SBATCH -n 12 # Se solicita n tareas
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH -t 02:00:00 


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
echo SLURM_MEM_PER_CPU: $SLURM_MEM_PER_CPU

#For creating folder for logs
LOGS='./logparallel'
CREATEFOLDER $LOGS

#Create parralel job
parallel="parallel --delay 2 -j $SLURM_NTASKS --joblog $LOGS/runtask_exploration.log --resume-failed"

#Create srun command
srun="srun --exclusive -N1 -n1 -c$SLURM_CPUS_PER_TASK --mem-per-cpu=$SLURM_MEM_PER_CPU"

#Loading modules
module load myqlm

#For saving DataFrames
FOLDER="/mnt/netapp1/Store_CESGA/home/cesga/gferro/Datos_2022_10_06/"
CREATEFOLDER $FOLDER
echo $FOLDER
FILENAME="RQAE_2022_10_06.csv"

#For looping
START=0
END=100

$parallel "$run python ../benchmark_ae_estimation_price.py  --RQAE --folder $FOLDER --name $FILENAME --save --exe -id {}" ::: $(seq $START $END)

#We can use an input file for looping staff
#cat ./txt_file.txt | $parallel "$srun ./Wraper.sh {}" 
#parallel --delay .2 -j $SLURM_NTASKS --joblog $LOGS/runtask_exploration.log --resume-failed srun --exclusive -N1 -n1 -c$SLURM_CPUS_PER_TASK --mem-per-cpu=$SLURM_MEM_PER_CPU python ./benchmark_ae_estimation_price.py  --RQAE --folder $FOLDER --name $FILENAME --save --exe -id {} ::: $(seq 0 127)
