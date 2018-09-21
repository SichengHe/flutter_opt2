import os
import argparse


# @copyright Sicheng He, Mon Mar 26 16:52:11 EDT 2018

# sample argument:
# python file_mover.py --adflowParDir "/home/sichenghe/hg" --petscParDir "/home/sichenghe/packages" --flag_complex2real False

#===========================================================
#      complex-real components convertor
#===========================================================

# assume we are converting a running version to either of complex or real version of the whole package of the code
# two types of OPERATION: 
#                         (I). running version (real) -> real, complex -> running version (flag_complex2real=True);
#                         (II). running version (complex) -> complex, complex -> running version (flag_complex2real=True).False
# In terms of each OPERATION (either (I) or (II)):
#                         adflow, pywarpustruct, petsc conversion (e.g. in (I) adflow -> adflow_real, adflow_complex -> adflow)


# get the directories
parser = argparse.ArgumentParser()
parser.add_argument("--adflowParDir", help="adflow parent dir",
                    type=str, default='/home/sichenghe/hg')
parser.add_argument("--petscParDir", help="petsc parent dir",
                   type=str, default='/home/sichenghe/packages')
parser.add_argument("--flag_complex2real", help="bool var for complex to real implementation. True or False",
                   type=str, default=True)

args = parser.parse_args()

adflowPar_dir = args.adflowParDir
pywarpustructpar_dir = adflowPar_dir # pywarpustruct should be under the same parent dir with adflow
petscPar_dir = args.petscParDir

flag_complex2real = args.flag_complex2real

if flag_complex2real == 'False':
    flag_complex2real = False
else:
    flag_complex2real = True

# preparison of file directories 
# if complex (real) 2 real (complex) then there should not be any real (complex) file before the conversion
if flag_complex2real:

    # active to inactive modules
    adflow_src1 = adflowPar_dir + '/' +'adflow'
    pywarpustruct_src1 = pywarpustructpar_dir + '/' +'pywarpustruct'
    petsc_src1 = petscPar_dir + '/' + 'petsc-3.6.1'

    adflow_tar1 = adflowPar_dir + '/' +'adflow_complex'
    pywarpustruct_tar1 = pywarpustructpar_dir + '/' +'pywarpustruct_complex'
    petsc_tar1 = petscPar_dir + '/' + 'petsc-3.6.1_complex'

    # inactive to active modules
    adflow_src2 = adflowPar_dir + '/' +'adflow_real'
    pywarpustruct_src2 = pywarpustructpar_dir + '/' +'pywarpustruct_real'
    petsc_src2 = petscPar_dir + '/' + 'petsc-3.6.1_real'

    adflow_tar2 = adflowPar_dir + '/' +'adflow'
    pywarpustruct_tar2 = pywarpustructpar_dir + '/' +'pywarpustruct'
    petsc_tar2 = petscPar_dir + '/' + 'petsc-3.6.1'


else:

    # active to inactive modules
    adflow_src1 = adflowPar_dir + '/' +'adflow'
    pywarpustruct_src1 = pywarpustructpar_dir + '/' +'pywarpustruct'
    petsc_src1 = petscPar_dir + '/' + 'petsc-3.6.1'

    adflow_tar1 = adflowPar_dir + '/' +'adflow_real'
    pywarpustruct_tar1 = pywarpustructpar_dir + '/' +'pywarpustruct_real'
    petsc_tar1 = petscPar_dir + '/' + 'petsc-3.6.1_real'

    # inactive to active modules
    adflow_src2 = adflowPar_dir + '/' +'adflow_complex'
    pywarpustruct_src2 = pywarpustructpar_dir + '/' +'pywarpustruct_complex'
    petsc_src2 = petscPar_dir + '/' + 'petsc-3.6.1_complex'

    adflow_tar2 = adflowPar_dir + '/' +'adflow'
    pywarpustruct_tar2 = pywarpustructpar_dir + '/' +'pywarpustruct'
    petsc_tar2 = petscPar_dir + '/' + 'petsc-3.6.1'


src1_list = [adflow_src1, pywarpustruct_src1, petsc_src1]
tar1_list = [adflow_tar1, pywarpustruct_tar1, petsc_tar1]
src2_list = [adflow_src2, pywarpustruct_src2, petsc_src2]
tar2_list = [adflow_tar2, pywarpustruct_tar2, petsc_tar2]

# sanity check:
# # whether the file already exists or the file to be converted doesnt exist
file_conflict = os.path.isdir(adflow_tar1) or  os.path.isdir(pywarpustruct_tar1) or os.path.isdir(petsc_tar1)
file_conflict = file_conflict or ((not os.path.isdir(adflow_src1)) or  (not os.path.isdir(pywarpustruct_src1)) or (not os.path.isdir(petsc_src1)))
file_conflict = file_conflict or ((not os.path.isdir(adflow_src2)) or  (not os.path.isdir(pywarpustruct_src2)) or (not os.path.isdir(petsc_src2)))

if file_conflict:

    print("Error: file conflict!")
    exit()




# conversion
# 1
for i in xrange(len(src1_list)):

    command1 = 'mv ' + src1_list[i] + ' ' + tar1_list[i]
    os.system(command1)
    

# 2
for i in xrange(len(src2_list)):

    command2 = 'mv ' + src2_list[i] + ' ' + tar2_list[i]
    os.system(command2)



