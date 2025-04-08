import json
from network import Network
import pathlib

script_dir = pathlib.Path(__file__).parent.resolve()
json_file_name = 'best-network.json'
json_file_path = script_dir / json_file_name

with open(json_file_path, 'r') as f:
    json_data = f.read()


netw = Network.from_json(json_data)
loaded_as_json = json.dumps(netw.json())


if json_data.strip() == loaded_as_json.strip():
    print("Network loaded successfully and JSON data matches!")
else:
    print("Network loaded, but JSON data does not match!")