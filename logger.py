from loguru import logger
import sys
import os
import json
from dotenv import load_dotenv

class CustomLogger:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        self.environment = os.getenv("ENVIRONMENT", "prod")
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.logger = self.setup_logger() 

    def serialize(self, record) -> str:
        updatedRecord = record["extra"].copy()
        updatedRecord["function"] = record["function"]
        updatedRecord["module"] = record["module"]
        updatedRecord["line"] = record["line"]
        updatedRecord["file"] = record["file"].name
        updatedRecord["environment"] = self.environment

        if record["exception"]:
            updatedRecord["exception"] = str(record["exception"])

        return json.dumps(updatedRecord)

    def patching(self, record):
        if "extra" not in record:
            record["extra"] = {}
        # Update the extra field with serialized data
        record["extra"] = self.serialize(record)

    def setup_logger(self):
        env = self.environment
        log_level = self.log_level
        logger_ctx = logger.remove(0)  # Remove the default logger
        logger_ctx = logger.patch(self.patching)  # Ensure logger is patched before setup
        if env == "local":
            logger_ctx.add("app.log", level=log_level, format="<lvl>{level}</lvl>: {time:MM/DD/YYYY > HH:mm:ss!UTC} | {message} | {extra}", enqueue=True)
        else:  
            logger_ctx.add(sys.stdout, level=log_level, format="<lvl>{level}</lvl>: {time:MM/DD/YYYY > HH:mm:ss!UTC} | {message} | {extra}", enqueue=True)
        logger_ctx.info("Structured logger initialized")
        return logger_ctx

    def getLogger(self): 
        return self.logger

# Create a logger instance
logger_instance = CustomLogger().getLogger()


