import os
import subprocess

root_dir = '/scratch/clear/marbel/projects/bilevel_opt/data/outputs/multitask_cifar100'
out_file = 'metrics.json'
dir_nrs = [
    os.path.join(root_dir, d)
    for d in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, d)) and d.isdigit() and not os.path.isfile(os.path.join(root_dir, d,out_file))
]

print(len(dir_nrs))
print( dir_nrs )

# for Dir in dir_nrs:
#     name = os.path.join(Dir,'besteffort.sh')

#     launch_cmd = "oarsub -S" +  " -p \"not host like 'gpuhost25 gpuhost10'\" " +  name
#     # Launch job over SSH
#     subprocess.check_call(launch_cmd, shell=True)


