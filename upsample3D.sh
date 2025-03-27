#!/bin/bash

input_file=$1
path=$2

# Compute delta values
delta_x=$(3dinfo -di "$path/$input_file")
delta_y=$(3dinfo -dj "$path/$input_file")
delta_z=$(3dinfo -dk "$path/$input_file")

# Calculate scaled delta values
sdelta_x=$(echo "((sqrt($delta_x * $delta_x) / 4))" | bc -l)
sdelta_y=$(echo "((sqrt($delta_y * $delta_y) / 4))" | bc -l)
sdelta_z=$(echo "((sqrt($delta_z * $delta_z) / 4))" | bc -l)

# Print scaled deltas
echo "Scaled deltas:"
echo "X: $sdelta_x"
echo "Y: $sdelta_y"
echo "Z: $sdelta_z"

# Run 3dresample
3dresample -dxyz $sdelta_x $sdelta_y $sdelta_z -rmode NN -overwrite -prefix "$path/scaled_$input_file" -input "$path/$input_file"

#3dresample -dxyz 0.4 0.4 0.4 -rmode NN -prefix -overwrite scaled_rim -input rim.nii 

3dcalc -a "$path/scaled_$input_file" -datum short -gscale -expr 'a' -prefix "$path/scaled_$input_file" -overwrite
# 3dcalc -a scaled_rim+orig -datum short -gscale -expr 'a' -prefix scaled_rim -overwrite
# 3dAFNItoNIFTI scaled_rim+orig

echo "Processing completed. Output saved to: $path"