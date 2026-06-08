#!/bin/bash

# 蒸着品質分析システム - Bash 起動スクリプト
# 使用方法: chmod +x run_server.sh && ./run_server.sh

echo ""
echo "=========================================="
echo " 蒸着品質分析システム 起動"
echo "=========================================="
echo ""

# スクリプトの場所をカレントディレクトリに設定
cd "$(dirname "$0")"

# Python がインストール済みか確認
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "エラー: Python がインストールされていません"
    echo "以下のコマンドで Python をインストールしてください:"
    echo "  Ubuntu/Debian: sudo apt-get install python3"
    echo "  macOS: brew install python"
    exit 1
fi

# Python のバージョン確認
python3 --version || python --version

# 依存関係のインストール
echo ""
echo "依存関係を確認中..."
pip install -q streamlit pandas numpy plotly

# IPアドレスの取得
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    IP_ADDRESS=$(ipconfig getifaddr en0)
else
    # Linux
    IP_ADDRESS=$(hostname -I | awk '{print $1}')
fi

echo ""
echo "=========================================="
echo " 起動情報"
echo "=========================================="
echo "Streamlit アプリケーション起動中..."
echo ""
echo "ローカルアクセス:"
echo "  http://localhost:8501"
echo ""
echo "ネットワークアクセス:"
if [ -n "$IP_ADDRESS" ]; then
    echo "  http://$IP_ADDRESS:8501"
else
    echo "  ipconfig または hostname -I で IP アドレスを確認してください"
fi
echo ""
echo "=========================================="
echo ""

# Streamlit アプリを起動
streamlit run app.py
