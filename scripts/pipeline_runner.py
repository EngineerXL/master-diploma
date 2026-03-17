import json
import sys
from pathlib import Path
from src.dataset.pipeline import LidarOdometryPipeline
from tqdm import tqdm


def load_configs_from_json(json_file: str) -> list[dict]:
    """Load multiple pipeline configurations from a JSON file.

    The JSON file should contain an object with:
    - actor_settings: dict with voxel_size, points_max_distance, icp_max_iterations, algo
    - wrapper_settings: dict
    - config_name: str
    - ride_segments: array of objects with ride_info (ride_id, ride_frame_st, ride_frame_en)

    Example JSON structure:
    {
        "actor_settings": {...},
        "wrapper_settings": {...},
        "config_name": "test1",
        "ride_segments": [
            {"ride_info": {"ride_id": "...", "ride_frame_st": 0, "ride_frame_en": 100}},
            {"ride_info": {"ride_id": "...", "ride_frame_st": 100, "ride_frame_en": 200}}
        ]
    }

    Returns a list of configuration dictionaries.
    """
    path = Path(json_file)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    with open(path, "r") as f:
        data = json.load(f)

    # Must be an object (dict), not a list
    if not isinstance(data, dict):
        raise ValueError(
            "JSON file must contain an object with actor_settings, config_name, and ride_segments"
        )

    required_fields = [
        "actor_settings",
        "config_name",
        "ride_segments",
        "wrapper_settings",
    ]

    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field '{field}' in configuration")

    configs = []
    ride_segments = data["ride_segments"]

    for i, segment in enumerate(ride_segments):
        if not isinstance(segment, dict):
            raise ValueError(
                f"ride_segments[{i}] is not a valid segment object (expected dict)"
            )

        if "ride_info" not in segment:
            raise ValueError(
                f"Missing required field 'ride_info' in ride_segments[{i}]"
            )

        ride_info = segment["ride_info"]
        configs.append(
            {
                "ride_info": ride_info,
                "actor_settings": data["actor_settings"],
                "wrapper_settings": data["wrapper_settings"],
                "config_name": data["config_name"],
            }
        )

    return configs


def main():
    """Main entry point - accepts a JSON file with multiple pipeline configurations."""
    if len(sys.argv) < 2:
        print("Usage: python3 pipeline_runner.py <config.json>")
        print("")
        print("The JSON file must contain a list of pipeline configurations.")
        print("")
        print("Example:")
        print("   python3 pipeline_runner.py my_pipelines.json")
        sys.exit(1)

    json_file = sys.argv[1]
    configs = load_configs_from_json(json_file)

    # Process each pipeline configuration
    for config in tqdm(configs, desc="Processing pipelines..."):
        # for i, config in enumerate(configs, 1):
        pipeline = LidarOdometryPipeline(config=config)
        pipeline.process_ride()
        pipeline.save_results()

    print(f"\nAll {len(configs)} pipelines completed successfully!")


if __name__ == "__main__":
    main()
