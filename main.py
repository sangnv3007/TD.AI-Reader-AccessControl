from fastapi import FastAPI, File, UploadFile, status, Body
from pydantic import BaseModel
from typing import Union
from process import info_extraction_VNID
from fastapi.middleware.cors import CORSMiddleware
import json
import uvicorn
import os
import shutil
import base64
import time
import datetime
import asyncio
import fitz
# from thongke import create_connection, select_one_tasks, update_record, select_one_tasks_with_time, insert_record, select_time_last_record
from starlette import status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response, RedirectResponse
from starlette.types import ASGIApp

class Item(BaseModel):
    name: Union[str, None] = None
    stringbase64: Union[str, None] = None

class LimitUploadSize(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, max_upload_size: int) -> None:
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method == 'POST':
            if 'content-length' not in request.headers:
                return Response(status_code=status.HTTP_411_LENGTH_REQUIRED)
            content_length = int(request.headers['content-length'])
            if content_length > self.max_upload_size:
                return Response(status_code=status.HTTP_431_REQUEST_HEADER_FIELDS_TOO_LARGE)
        return await call_next(request)

app = FastAPI(
    title="TD.AI_Reader",
    description="Copyright 2024 for TAN DAN ., JSC. All right reserved\n",
    version="beta-0.0.1"
    )

origins = [
    "http://192.168.2.70:3011",
    "http://dangkyvaora.hanhchinhcong.org",
    "https://dangkyvaora.megasolution.vn",
    "https://quanlyvaora.megasolution.vn",
    "http://localhost:3012"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(LimitUploadSize, max_upload_size=10000000)  # ~10MB


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

@app.post("/IdentityCard/upload")
async def uploadFile(file: UploadFile = File(...)):
    # try:
    print(f'TD.AIReader IdentityCard INFO: {datetime.datetime.now()}')
    folder_save = 'files/IdentityCard/'
    pathSave = os.path.join(os.getcwd(),folder_save)
    os.makedirs(pathSave, exist_ok=True)
    with open(f'{folder_save}/{file.filename}', 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    return await info_extraction_VNID(f'./{folder_save}/{file.filename}')
    # except Exception as e:
    #     save_directory = os.getcwd() + '/error_files'
    #     os.makedirs(save_directory,exist_ok=True)
    #     with open(f'error_files/{file.filename}', 'wb') as buffer:
    #         shutil.copyfileobj(file.file, buffer)
    #     rs = {
    #             "errorCode": 3,
    #             "errorMessage": str(e),
    #             "results": []
    #         }
    #     return rs

# if __name__ == "__main__":
#     uvicorn.run(app,host='192.168.2.167', port=8006)
