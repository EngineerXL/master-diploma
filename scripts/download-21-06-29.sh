#!/bin/bash 
root=/data/boreas
aws s3 sync s3://boreas/boreas-2021-06-29-18-53 $root/boreas-2021-06-29-18-53 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request