#!/bin/bash 
root=/data/boreas
aws s3 sync s3://boreas/boreas-2021-09-09-15-28 $root/boreas-2021-09-09-15-28 --exclude "*"  --include "lidar/*" --include "applanix/*" --include "calib/*" --no-sign-request