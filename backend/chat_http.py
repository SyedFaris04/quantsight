"""Prepare the generated answer before committing successful stream headers."""
from itertools import chain
import logging

from fastapi.responses import JSONResponse, StreamingResponse
from chatbot_engine import ChatServiceError


def chat_response(stream):
    try:
        first = next(stream)
    except ChatServiceError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": {
            "code": exc.code, "message": str(exc), "retryable": exc.retryable,
        }})
    except Exception as exc:
        logging.getLogger("nuroquant-api").error("Chat failed: %s", type(exc).__name__)
        return JSONResponse(status_code=502, content={"detail": {
            "code": "chat_error", "message": "The AI Assistant could not complete this request. Please retry.",
            "retryable": True,
        }})
    return StreamingResponse(chain([first], stream), media_type="text/plain",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
