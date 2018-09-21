# @copyright Sicheng He, Mon Mar 26 16:52:11 EDT 2018

import io
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--complexPetscLine", help="complex petsc line number",
                    type=int, default=136) # notice this starts with 1!
parser.add_argument("--realPetscLine", help="complex petsc line number",
                    type=int, default=134)
parser.add_argument("--flag_complex2real", help="bool var for complex to real implementation. True or False",
                   type=str, default=True)

args = parser.parse_args()


ind_complex_line = args.complexPetscLine - 1
ind_real_line = args.realPetscLine - 1
flag_complex2real = args.flag_complex2real

if flag_complex2real == 'False':
    flag_complex2real = False
else:
    flag_complex2real = True

with open('/home/sichenghe/.bashrc', 'r') as file:

    data = file.readlines()

if flag_complex2real:

    data[ind_complex_line] = '# ' + data[ind_complex_line]
    data[ind_real_line] = data[ind_real_line][2:]

else:

    data[ind_real_line] = '# ' + data[ind_real_line]
    data[ind_complex_line] = data[ind_complex_line][2:]


with open('/home/sichenghe/.bashrc', 'w') as file:

    file.writelines(data)





