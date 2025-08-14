from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from logger import logger_instance
import uuid

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        client_ip = request.client.host
        path = request.url.path

        logger_ctx = logger_instance
        logger_ctx = logger_ctx.bind(request_id=request_id, client_ip=client_ip, path=path)
        logger_ctx.info("Incoming request")

        try:
            response = await call_next(request)
            logger_ctx.info("Request completed")
            return response
        except Exception as e:
            logger_ctx.error(f"Unhandled exception during request: {str(e)}")
            raise