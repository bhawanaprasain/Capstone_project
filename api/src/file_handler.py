import json

def read_json_file(file_path):
    """
    Read data from a JSON file.

    Parameters:
    - file_path (str): The path to the JSON file.

    Returns:
    - dict: The parsed JSON data.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File not found at path '{file_path}'")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
