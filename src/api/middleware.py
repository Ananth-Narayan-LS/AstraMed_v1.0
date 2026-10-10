
import logging
import re
import uuid

from starlette.middleware.base import BaseHTTPMiddleware


logger = logging.getLogger(__name__)
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,100}$")


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        supplied_id = request.headers.get("X-Request-ID", "")

        request_id = (
            supplied_id
            if REQUEST_ID_PATTERN.fullmatch(supplied_id)
            else str(uuid.uuid4())
        )

        request.state.request_id = request_id

        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "Unhandled request failure: request_id=%s",
                request_id,
            )
            raise

        response.headers["X-Request-ID"] = request_id
        return response
