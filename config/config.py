import os


REQUIRED_ENV_VARS = [
    "PINECONE_API_KEY",
    "PINECONE_ENVIRONMENT",
    "OPENAI_API_KEY",
    "OPENAI_ORG_KEY",
]


class Config:
    def __init__(self):
        self.OPENAI_API_KEY = self.__get("OPENAI_API_KEY")
        self.OPENAI_ORG_KEY = self.__get("OPENAI_ORG_KEY")
        self.PINECONE_API_KEY = self.__get("PINECONE_API_KEY")
        self.PINECONE_ENVIRONMENT = self.__get("PINECONE_ENVIRONMENT")

        self.__raise_if_missing_required_env_vars()

    def __raise_if_missing_required_env_vars(self):
        for var in REQUIRED_ENV_VARS:
            if not getattr(self, var):
                raise ValueError(f"Missing required environment variable: {var}")

    def __get(self, key, default=None) -> str:
        val = os.environ.get(key, default=default)

        if val is not None:
            return val
        else:
            raise ValueError(f"Missing required environment variable: {key}")