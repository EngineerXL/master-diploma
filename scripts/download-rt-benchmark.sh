#!/bin/bash 
root=/data/boreas
aws s3 sync s3://boreas/boreas-2025-01-08-11-22 $root/boreas-2025-01-08-11-22 --exclude "*"  --include "lidar/*" --include "imu/*" --include "applanix/*" --include "calib/*" --no-sign-request
aws s3 sync s3://boreas/boreas-2024-12-04-12-08 $root/boreas-2024-12-04-12-08 --exclude "*"  --include "lidar/*" --include "imu/*" --include "applanix/*" --include "calib/*" --no-sign-request
aws s3 sync s3://boreas/boreas-2024-12-04-14-50 $root/boreas-2024-12-04-14-50 --exclude "*"  --include "lidar/*" --include "imu/*" --include "applanix/*" --include "calib/*" --no-sign-request
aws s3 sync s3://boreas/boreas-2024-12-23-16-27 $root/boreas-2024-12-23-16-27 --exclude "*"  --include "lidar/*" --include "imu/*" --include "applanix/*" --include "calib/*" --no-sign-request
aws s3 sync s3://boreas/boreas-2025-02-15-17-19 $root/boreas-2025-02-15-17-19 --exclude "*"  --include "lidar/*" --include "imu/*" --include "applanix/*" --include "calib/*" --no-sign-request
aws s3 sync s3://boreas/boreas-2024-12-04-15-19 $root/boreas-2024-12-04-15-19 --exclude "*"  --include "lidar/*" --include "imu/*" --include "applanix/*" --include "calib/*" --no-sign-request
aws s3 sync s3://boreas/boreas-2024-12-04-12-34 $root/boreas-2024-12-04-12-34 --exclude "*"  --include "lidar/*" --include "imu/*" --include "applanix/*" --include "calib/*" --no-sign-request
aws s3 sync s3://boreas/boreas-2024-12-23-16-44 $root/boreas-2024-12-23-16-44 --exclude "*"  --include "lidar/*" --include "imu/*" --include "applanix/*" --include "calib/*" --no-sign-request
aws s3 sync s3://boreas/boreas-2025-07-18-15-30 $root/boreas-2025-07-18-15-30 --exclude "*"  --include "lidar/*" --include "imu/*" --include "applanix/*" --include "calib/*" --no-sign-request
aws s3 sync s3://boreas/boreas-2025-08-13-09-01 $root/boreas-2025-08-13-09-01 --exclude "*"  --include "lidar/*" --include "imu/*" --include "applanix/*" --include "calib/*" --no-sign-request