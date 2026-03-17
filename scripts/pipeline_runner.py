import json
import sys
from pathlib import Path
from src.dataset.pipeline import LidarOdometryPipeline
from tqdm import tqdm


def load_configs_from_json(json_file: str) -> list[dict]:
    """Load multiple pipeline configurations from a JSON file.
    
    The JSON file should contain a list of configuration objects, each with:
    - ride_info: dict with ride_id, ride_frame_st, ride_frame_en
    - actor_settings: dict with voxel_size, points_max_distance, icp_max_iterations, algo
    - config_name: str
    
    Example JSON structure:
    [
        {
            "ride_info": {"ride_id": "...", "ride_frame_st": 0, "ride_frame_en": 100},
            "actor_settings": {...},
            "config_name": "test1"
        },
        {
            "ride_info": {"ride_id": "...", "ride_frame_st": 100, "ride_frame_en": 200},
            "actor_settings": {...},
            "config_name": "test2"
        }
    ]
    
    Returns a list of configuration dictionaries.
    """
    path = Path(json_file)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file}")
    
    with open(path, "r") as f:
        data = json.load(f)
    
    # Must be a list of configurations
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of pipeline configurations")
    
    configs = []
    required_fields = ["ride_info", "actor_settings", "config_name"]
    
    for i, config in enumerate(data):
        if not isinstance(config, dict):
            raise ValueError(f"Item {i} is not a valid configuration object (expected dict)")
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in configuration item {i}")
        
        configs.append(config)
    
    return configs


def main():
    """Main entry point - accepts a JSON file with multiple pipeline configurations."""
    if len(sys.argv) < 2:
        print("Usage: python3 pipeline_runner.py <config.json>")
        print("")
        print("The JSON file must contain a list of pipeline configurations.")
        print("")
        print("Example:")
        print('   python3 pipeline_runner.py my_pipelines.json')
        sys.exit(1)
    
    json_file = sys.argv[1]
    configs = load_configs_from_json(json_file)
    
    # Process each pipeline configuration
    for config in tqdm(configs, desc = "Processing pipelines..."):
    # for i, config in enumerate(configs, 1):
        pipeline = LidarOdometryPipeline(config=config)
        pipeline.process_ride()
        pipeline.save_results()
    
    print(f"\nAll {len(configs)} pipelines completed successfully!")


if __name__ == "__main__":
    main()
