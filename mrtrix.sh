#!/bin/bash
# @author Bob Dougherty

if [ -z "$1" ]; then
   echo
   echo "Usage: mrtrix.sh data_dir out_dir"
   echo
   echo "Run a mrTrix ACT processing stream on data.nii.gz in data_dir. Based on http://mrtrix.readthedocs.io/en/latest/tutorials/hcp_connectome.html."
   echo
   echo "For meore details, see: "
   echo
   echo "Smith, R. E., Tournier, J.-D., Calamante, F., & Connelly, A. (2012). Anatomically-constrained tractography: Improved diffusion MRI streamlines tractography through effective use of anatomical information. NeuroImage, 62(3), 1924–1938. doi:10.1016/j.neuroimage.2012.06.005."
   echo
   exit
fi

# Note: adjust the following to suit your system
fsdir="/software/freesurfer/v5.3.0"
mrtrixdir="/home/bobd/git/mrtrix3"
# Set the initial number of streamlines (e.g., 20M)
num_sl="20000000"
# Set the target number of streamlines after SIFTing (e.g., 2M)
num_sift="2000000"

data=$1
out=$2

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${mrtrixdir}/lib

if [ -e ${out}/DWI.mif ]; then
  echo "Using existing ${out}/DWI.mif file.."
else
  dwiextract ${data}/data.nii.gz - -fslgrad ${data}/bvecs ${data}/bvals -shell 0,3000 | mrconvert - ${out}/DWI.mif -datatype float32 -stride 0,0,0,1;
  dwiextract ${out}/DWI.mif - -bzero | mrmath - mean ${out}/meanb0.mif -axis 3;
fi
if [ -e ${out}/response.txt ]; then
  echo "Using existing ${out}/response.txt file.."
else
  maskfilter ${data}/nodif_brain_mask.nii.gz erode - -npass 6 | dwi2response ${out}/DWI.mif ${out}/response.txt -mask - -lmax 8 -sf ${out}/sf_mask.mif;
fi
if [ -e ${out}/FOD.mif ]; then
  echo "Using existing ${out}/FOD.mif file.."
else
  dwi2fod ${out}/DWI.mif ${out}/response.txt ${out}/FOD.mif -mask ${data}/nodif_brain_mask.nii.gz;
  mrresize ${out}/FOD.mif ${out}/FOD_downsampled.mif -scale 0.5 -interp sinc
fi
if [ -e ${out}/5TT.mif ]; then
  echo "Using existing ${out}/5TT.mif file.."
else
  ${mrtrixdir}/scripts/act_anat_prepare_fsl ${data}/../T1w_acpc_dc_restore.nii.gz ${out}/5TT.mif;
fi
if [ -e ${out}/nodes_fixSGM.mif ]; then
  echo "Using existing ${out}/nodes_fixSGM.mif file.."
else
  labelconfig ${data}/../aparc+aseg.nii.gz ${mrtrixdir}/src/connectome/config/fs_default.txt ${out}/nodes.mif -lut_freesurfer ${fsdir}/FreeSurferColorLUT.txt
  ${mrtrixdir}/scripts/fs_parc_replace_sgm_first ${out}/nodes.mif ${data}/../T1w_acpc_dc_restore.nii.gz ${mrtrixdir}/src/connectome/config/fs_default.txt ${out}/nodes_fixSGM.mif
fi
if [ -e ${out}/20M.tck ]; then
  echo "Using existing ${out}/20M.tck file.."
else
  tckgen ${out}/FOD.mif ${out}/20M.tck -act ${out}/5TT.mif -downsample 6 -backtrack -crop_at_gmwmi -seed_dynamic ${out}/FOD.mif -maxlength 250 -number ${num_sl}
fi
if [ -e ${out}/2M_SIFT.tck ]; then
  echo "Using existing ${out}/2M_SIFT.tck file.."
else
  tcksift ${out}/20M.tck ${out}/FOD_downsampled.mif ${out}/2M_SIFT.tck -act ${out}/5TT.mif -term_number ${num_sift}
  rm -f ${out}/20M.tck
fi

