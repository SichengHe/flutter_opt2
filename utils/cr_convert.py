# @copyright Sicheng He, Mon Mar 26 16:52:11 EDT 2018
# make sure complexPetscLine is 136 and realPetscLine 134!
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--flag_complex2real", help="bool var for complex to real implementation. True or False",
                   type=str, default=True)

args = parser.parse_args()

flag_complex2real = args.flag_complex2real
if flag_complex2real == 'False':
    flag_complex2real = False
else:
    flag_complex2real = True

if flag_complex2real: str_flag_complex2real = 'True'
if not flag_complex2real: str_flag_complex2real = 'False'
    
command1 = 'python file_mover.py --adflowParDir "/home/sichenghe/hg" --petscParDir "/home/sichenghe/packages" --flag_complex2real ' + str_flag_complex2real
command2 = 'python file_editor.py --complexPetscLine 136 --realPetscLine 134 --flag_complex2real ' + str_flag_complex2real

os.system(command1)
os.system(command2)

