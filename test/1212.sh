#!/bin/bash						
#DSUB -n 1211_job					
#DSUB -N 4						
#DSUB -A root.huadjyin.team_a			
#DSUB -R "cpu=128;gpu=4;mem=240000"		
#DSUB -oo %J.out					
#DSUB -eo %J.err					


source /home/HPCBase/tools/module-5.2.0/init/profile.sh         
# module purge 
module use /home/HPCBase/modulefiles/					#--加载软件列表文件
module load apps/abinit/8.10.3_kgcc9.3.1_hmpi1.2.0 
#abinit --version 										#--计算执行命令，对应在普通

source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh		#--加载module工具
conda activate SAAMS                    #--加载conda环境
source /home/HPCBase/tools/module-5.2.0/init/profile.sh && module purge && module use /home/HPCBase/modulefiles/
module load libs/openblas/0.3.18_kgcc9.3.1 compilers/cuda/11.8.0 compilers/kgcc/9.3.1			
python /home/share/huadjyin/home/zhoutao3/SAAMS/test/test.py
