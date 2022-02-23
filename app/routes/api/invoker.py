from fastapi import APIRouter, File, UploadFile, Body
import app.controllers.invoker_controller as controller_invoker

# Prefix for scores endpoints.
PREFIX = "invoker"
# Tags for docs.
TAGS = ["Invoker"]

router = APIRouter()


@router.get("/")
async def index():
    return {
        "message": PREFIX.upper(),
    }


@router.post("/")
async def invoke(invoice_file: UploadFile):
    return controller_invoker.get_invoke_response(invoice_file)
