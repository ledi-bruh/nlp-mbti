from pydantic_settings import BaseSettings, SettingsConfigDict


__all__ = ['settings']


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file='.env'
    )

    nlp_model_name: str
    tg_bot_token: str


settings = Settings()
