#!/bin/sh

# SBATCH --ntasks-per-node=3
# SBATCH --cpus-per-task=64
# SBATCH --mem-per-cpu=3G
# SBATCH -n 1# Se solicita n tareas
# SBATCH -t 20:00:00 
# SBATCH --array=0-27
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


FOLDER="/home/gferro/Datos/"
CREATEFOLDER $FOLDER
echo $FOLDER

FILENAME="Todo.csv"
python quantum_integration_test.py --CQPEAE --folder $FOLDER --all --name $FILENAME --save --exe & 
python quantum_integration_test.py --IQPEAE --folder $FOLDER --all --name $FILENAME --save --exe & 
#python quantum_integration_test.py --IQAE --folder $FOLDER --all --name $FILENAME --save --exe & 
#python quantum_integration_test.py --RQAE --folder $FOLDER --all --name $FILENAME --save --exe & 
#python quantum_integration_test.py --MLAE --folder $FOLDER --all --name $FILENAME --save --exe & 
