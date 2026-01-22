@echo off
chcp 65001
title DEM - Deep Energy Method (Streamlit)

REM 切换到当前脚本所在目录
cd /d %~dp0



REM 关闭 Streamlit 自动弹浏览器的欢迎信息
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

REM 启动程序
streamlit run DEM.py --server.headless=false

pause
