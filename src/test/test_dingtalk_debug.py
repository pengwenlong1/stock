# -*- coding: utf-8 -*-
"""
钉钉签名调试脚本

逐步打印签名计算过程，找出问题所在。
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

with open(settings_path, 'r', encoding='utf-8') as f:
    settings = json.load(f)

webhook = settings.get('dingtalk_webhook', '')
secret = settings.get('dingtalk_secret', '')

print("=" * 60)
print("钉钉签名调试")
print("=" * 60)
print(f"Webhook: {webhook}")
print(f"Secret: {secret}")
print()

# 步骤1：生成时间戳
timestamp = str(round(time.time() * 1000))
print(f"步骤1 - 时间戳: {timestamp}")

# 步骤2：构造签名字符串
string_to_sign = timestamp + '\n' + secret
print(f"步骤2 - 签名字符串: {repr(string_to_sign)}")

# 步骤3：HmacSHA256 计算
hmac_code = hmac.new(
    secret.encode('utf-8'),
    string_to_sign.encode('utf-8'),
    digestmod=hashlib.sha256
).digest()
print(f"步骤3 - HMAC结果(bytes): {hmac_code}")
print(f"步骤3 - HMAC结果(hex): {hmac_code.hex()}")

# 步骤4：Base64 编码
b64_bytes = base64.b64encode(hmac_code)
print(f"步骤4 - Base64(bytes): {b64_bytes}")
b64_str = b64_bytes.decode('utf-8')
print(f"步骤4 - Base64(string): {b64_str}")

# 步骤5：URL 编码
sign = urllib.parse.quote_plus(b64_str)
print(f"步骤5 - URL编码签名: {sign}")

# 步骤6：构造完整URL
url = webhook + '&timestamp=' + timestamp + '&sign=' + sign
print(f"步骤6 - 完整URL: {url}")

# 步骤7：发送请求
print()
print("=" * 60)
print("发送测试消息")
print("=" * 60)

headers = {'Content-Type': 'application/json;charset=utf-8'}
data = {
    "msgtype": "text",
    "text": {
        "content": "【测试】钉钉签名调试测试"
    }
}

try:
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers=headers
    )
    response = urllib.request.urlopen(req, timeout=10)
    result = json.loads(response.read().decode('utf-8'))
    print(f"响应结果: {result}")

    if result.get('errcode') == 0:
        print("\n✅ 发送成功！")
    else:
        print(f"\n❌ 发送失败！")
        print(f"错误码: {result.get('errcode')}")
        print(f"错误信息: {result.get('errmsg')}")

except Exception as e:
    print(f"请求异常: {e}")