@echo off
chcp 65001 >nul
title He Thong Phat Hien Gian Lan Ngan Hang
echo ============================================================
echo   DANG KHOI DONG UNG DUNG PHAT HIEN GIAN LAN NGAN HANG...
echo   (Vui long giu cua so den nay trong suot qua trinh su dung)
echo ============================================================

:: Chay server ngam
start "" /B streamlit run app.py --server.port 8501 --server.headless true

:: Cho 3 giay de he thong nap thu vien
timeout /t 3 /nobreak >nul

:: Mo giao dien duoi dang cua so App doc lap (khong thanh dia chi trinh duyet)
start msedge --app=http://localhost:8501 || start chrome --app=http://localhost:8501 || start http://localhost:8501