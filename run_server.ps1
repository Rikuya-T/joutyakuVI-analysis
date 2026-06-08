# 蒸着品質分析システム - PowerShell 起動スクリプト
# 使用方法: PowerShell で実行するか、ダブルクリックで実行

Write-Host "==========================================" -ForegroundColor Green
Write-Host " 蒸着品質分析システム 起動" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""

# スクリプトの場所をカレントディレクトリに設定
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

# Python がインストール済みか確認
try {
    $pythonVersion = python --version 2>$null
    Write-Host "Python 確認: $pythonVersion" -ForegroundColor Cyan
} catch {
    Write-Host "エラー: Python がインストールされていません" -ForegroundColor Red
    Write-Host "python --version を実行して確認してください" -ForegroundColor Red
    Read-Host "エンターキーを押して終了"
    exit 1
}

# 依存関係のインストール
Write-Host ""
Write-Host "依存関係を確認中..." -ForegroundColor Yellow
pip install -q streamlit pandas numpy plotly

# IPアドレスの取得
$ipAddress = (Get-NetIPAddress -AddressFamily IPv4 -AddressState Preferred | Where-Object { $_.IPAddress -notmatch "^127\." }).IPAddress

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host " 起動情報" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Streamlit アプリケーション起動中..." -ForegroundColor Green
Write-Host ""
Write-Host "ローカルアクセス:" -ForegroundColor Cyan
Write-Host "  http://localhost:8501"
Write-Host ""
Write-Host "ネットワークアクセス:" -ForegroundColor Cyan
if ($ipAddress) {
    Write-Host "  http://$ipAddress`:8501"
}
Write-Host "  ※ 複数IPがある場合は ipconfig コマンドで確認してください"
Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""

# Streamlit アプリを起動
streamlit run app.py

Read-Host "エンターキーを押して終了"
