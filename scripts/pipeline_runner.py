import os
import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.dataset.pipeline import LidarOdometryPipeline


def main():
    pipeline = LidarOdometryPipeline(
        {
            "ride_info": {
                "ride_id": "boreas-2021-05-06-13-19",
                "ride_frame_st": 0,
                "ride_frame_en": 500,
            },
            "actor_settings": {
                "voxel_size": 1,
                "points_max_distance": 50,
                "icp_max_iterations": 100,
                "algo": "icp_geman_mcclure",
            },
            "config_name": "test",
        }
    )
    pipeline.process_ride()
    output_path = pipeline.save_results()
    print(f"Finished pipeline: {output_path}")


if __name__ == "__main__":
    main()
