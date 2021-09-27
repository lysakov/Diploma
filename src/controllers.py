import json

class JsonAuthController(object):

    def __init__(self, file_path : str) -> None:
        json_str = None
        with open(file_path, "r") as auth_file:
            json_str = auth_file.read()

        self.token = json.loads(json_str)

    def get_token(self) -> str:
        return self.token["token"]