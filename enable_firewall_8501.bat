@echo off
REM ================================================
REM  Windowsファイアウォール ポート8501 開放スクリプト
REM  ※ 管理者として実行してください
REM ================================================

echo.
echo ==========================================
echo  ファイアウォール設定 (ポート8501)
echo ==========================================
echo.
echo このスクリプトは管理者権限が必要です。
echo.

REM 管理者権限の確認
net session >nul 2>&1
if errorlevel 1 (
    echo エラー: 管理者権限がありません。
    echo このファイルを右クリック → "管理者として実行" してください。
    echo.
    pause
    exit /b 1
)

echo [実行中] ファイアウォールルールを追加中...
netsh advfirewall firewall add rule name="StreamlitApp8501" dir=in action=allow protocol=TCP localport=8501

if errorlevel 1 (
    echo.
    echo [警告] ルールの追加に失敗しました。すでに存在する可能性があります。
) else (
    echo.
    echo [完了] ファイアウォールルールを追加しました。
    echo   ルール名: StreamlitApp8501
    echo   ポート: TCP 8501
    echo   方向: 受信
)

echo.
echo 設定後は run_server.bat を起動し、
echo 同一ネットワーク内の他PCから以下でアクセスしてください:
echo   http://（このPCのIPアドレス）:8501
echo.
echo ※ IPアドレスは ipconfig コマンドで確認できます
echo.
pause
