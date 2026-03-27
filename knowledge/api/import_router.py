import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


def create_app()->FastAPI:
    """
    负责创建fastapi实例
    Returns:

    """
    # 1. 实例化fastapi实例
    app = FastAPI(description="知识库导入")

    # 2. 跨域配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
    )

    # 3.将静态资源的目录挂载到app上
    front_page_dir = get_front_page_dir()


def register_router(app:FastAPI):

    # 1.处理导入页面访问请求
    @app.get('./import')
    def import_root():




if __name__ == '__main__':
    uvicorn.run(app="",port=8000)
