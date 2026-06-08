@echo off
REM 蒸着品質分析システム - 起動スクリプト
REM このスクリプトはStreamlitアプリをネットワークに公開します

echo.
echo ==========================================
echo  蒸着品質分析システム 起動
echo ==========================================
echo.

REM カレントディレクトリをスクリプトの場所に変更
cd /d "%~dp0"

REM Python環境の確認
python --version >nul 2>&1
if errorlevel 1 (
    echo エラー: Python がインストールされていません
    pause
    exit /b 1
)

REM 依存関係のインストール確認
echo 依存関係を確認中...
pip install -q streamlit pandas numpy plotly

REM IPアドレスの取得と表示
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /C:"IPv4"') do (
    set IP=%%a
    goto :show_ip
)
:show_ip
set IP=%IP: =%

echo.
echo ==========================================
echo  起動情報
echo ==========================================
echo   ローカルアクセス: http://localhost:8501
echo   LAN アクセス    : http://%IP%:8501
echo ==========================================
echo.
echo ※ LANアクセスできない場合は enable_firewall_8501.bat を
echo   管理者として実行してからやり直してください。
echo.

REM Streamlitアプリを起動
streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true

pause
