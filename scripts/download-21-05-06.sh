#!/bin/bash 
root=/data/boreas
aws s3 sync s3://boreas/boreas-2021-05-06-13-19 $root/boreas-2021-05-06-13-19 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request
