import uvicorn
import config.environment as config_environment

if __name__ == "__main__":
    uvicorn.run("app:application", host=config_environment.get_host(), port=config_environment.get_port(), reload=config_environment.is_dev())
