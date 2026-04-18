import json

METADATA_PATH = "models/metadata.json"

def get_active_model_version():
    with open(METADATA_PATH) as f:
        data = json.load(f)
    return data["active_version"]
