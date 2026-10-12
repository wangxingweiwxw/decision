import json
import os
import sys

import requests


def main():
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    workspace_id = os.environ.get("BAILIAN_WORKSPACE_ID", "").strip()
    if not api_key or not workspace_id:
        print(
            "请先设置环境变量 DASHSCOPE_API_KEY 和 BAILIAN_WORKSPACE_ID。",
            file=sys.stderr,
        )
        return 1

    url = (
        f"https://{workspace_id}.cn-beijing.maas.aliyuncs.com"
        "/compatible-mode/v1/systemone"
    )
    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "decision-model-preview",
                "state": "我昨天买的会员一直没有开通，请把钱退给我。",
                "questions": {
                    "intent": {
                        "type": "choice",
                        "instructions": "判断用户的主要诉求。",
                        "criteria": {
                            "refund": "要求退款",
                            "consult": "咨询产品或使用方法",
                            "other": "其他诉求",
                        },
                    },
                    "needs_refund": {
                        "type": "noul",
                        "instructions": "用户是否明确要求退款？",
                    },
                },
            },
            timeout=60,
        )
    except requests.RequestException as exc:
        print(f"网络请求失败：{exc}", file=sys.stderr)
        return 1

    if not response.ok:
        print(
            f"调用失败，HTTP {response.status_code}：{response.text}",
            file=sys.stderr,
        )
        return 1

    try:
        result = response.json()
    except ValueError:
        print("调用失败：服务器未返回有效 JSON。", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())