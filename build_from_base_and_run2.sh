#!/bin/bash
# 用安装好pip库的镜像构建项目镜像并运行

# 构建Docker镜像并设置tag为当前时间戳
docker build -t tracegra-be:latest -f Dockerfile-backend .

# 运行Docker容器，将宿主机的8080端口映射到容器内的5000端口
docker run -d -p 8181:5000 --name my-tracegra-be tracegra-be:latest
