# -*- coding: utf-8 -*-
"""
钉钉告警测试脚本

用于验证钉钉机器人配置是否正确。
"""
import json
import hmac
import hashlib
import base64
import urllib.parse
import urllib.request
import time
from pathlib import Path

# 加载配置
project_root = Path(__file__).resolve().parent.parent.parent
settings_path = project_root / 'config' / 'settings.json'

print(f"配置文件路径: {settings_path}")

with open(settings_path, 'r', encoding='utf-8') as f:
    settings = json.load(f)

webhook = settings.get('dingtalk_webhook', '')
secret = settings.get('dingtalk_secret', '')

print(f"Webhook: {webhook}")
print(f"Secret: {secret[:10]}...（已截断）")

# 提取 access_token
if 'access_token=' in webhook:
    access_token = webhook.split('access_token=')[1].split('&')[0]
    print(f"Access Token: {access_token}")
else:
    print("无法从 webhook 中提取 access_token")

# 测试发送
def send_test_message():
    """发送测试消息"""
    timestamp = int(time.time() * 1000)

    # 生成签名
    string_to_sign = f"{timestamp}\n{secret}"
    hmac_code = hmac.new(
        secret.encode('utf-8'),
        string_to_sign.encode('utf-8'),
        digestmod=hashlib.sha256
    ).digest()
    # 先 base64 编码，再 decode 成字符串，最后 URL 编码
    sign = urllib.parse.quote_plus(base64.b64encode(hmac_code).decode('utf-8'))

    url = f"{webhook}&timestamp={timestamp}&sign={sign}"

    headers = {'Content-Type': 'application/json;charset=utf-8'}
    data = {
        "msgtype": "text",
        "text": {
            "content": "【测试消息】钉钉机器人配置验证成功！"
        },
        "at": {
            "atMobiles": [],
            "isAtAll": False
        }
    }

    print(f"\n请求URL: {url[:100]}...（已截断）")

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers=headers
        )
        response = urllib.request.urlopen(req, timeout=10)
        result = json.loads(response.read().decode('utf-8'))

        print(f"\n响应结果: {result}")

        if result.get('errcode') == 0:
            print("\n✅ 钉钉发送成功！配置正确。")
            return True
        else:
            print(f"\n❌ 钉钉发送失败！错误码: {result.get('errcode')}, 错误信息: {result.get('errmsg')}")
            return False

    except Exception as e:
        print(f"\n❌ 发送异常: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("钉钉告警配置测试")
    print("=" * 60)

    if not webhook or not secret:
        print("❌ 配置不完整，请检查 settings.json")
    else:
        send_test_message()