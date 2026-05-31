#!/usr/bin/env python3
"""Daily vLLM multimodal digest — fetches multimodal PR/issue activity from
vllm-project/vllm and creates a GitHub issue in the fork as a digest."""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from urllib.parse import quote

TOKEN = os.environ.get("GH_TOKEN", "")
FORK_OWNER = "skyloevil"
FORK_REPO = "vllm"
UPSTREAM = "vllm-project/vllm"
CST = timezone(timedelta(hours=8))
DIGEST_LABEL = "daily-digest"


def _request(method: str, url: str, payload: dict | None = None) -> dict | list:
    data = json.dumps(payload).encode() if payload else None
    req = Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"HTTP {e.code} for {url}: {body[:300]}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"Request error for {url}: {e}", file=sys.stderr)
        return {}


def search(query: str, per_page: int = 15) -> list[dict]:
    url = (
        f"https://api.github.com/search/issues"
        f"?q={quote(query)}&sort=updated&order=desc&per_page={per_page}"
    )
    result = _request("GET", url)
    return result.get("items", []) if isinstance(result, dict) else []


def ensure_label() -> None:
    base = f"https://api.github.com/repos/{FORK_OWNER}/{FORK_REPO}/labels"
    existing = _request("GET", f"{base}/{DIGEST_LABEL}")
    if existing.get("name"):
        return
    _request("POST", base, {"name": DIGEST_LABEL, "color": "0075ca",
                             "description": "Auto-generated daily digest"})


def create_issue(title: str, body: str) -> str:
    ensure_label()
    url = f"https://api.github.com/repos/{FORK_OWNER}/{FORK_REPO}/issues"
    result = _request("POST", url, {"title": title, "body": body,
                                    "labels": [DIGEST_LABEL]})
    return result.get("html_url", "(no URL)")


def fmt(items: list[dict], max_items: int = 12) -> str:
    if not items:
        return "_暂无相关内容_"
    lines = []
    for item in items[:max_items]:
        num = item["number"]
        title = item["title"][:78]
        user = item["user"]["login"]
        labels = " ".join(f'`{l["name"]}`' for l in item.get("labels", [])[:4])
        updated = item["updated_at"][:10]
        url = item["html_url"]
        assignee = f" 👤{item['assignee']['login']}" if item.get("assignee") else ""
        lines.append(
            f"- [#{num}]({url}) **{title}**  \n"
            f"  @{user} · {updated}{assignee} {labels}"
        )
    return "\n".join(lines)


def main() -> None:
    if not TOKEN:
        sys.exit("GH_TOKEN environment variable is required")

    today = datetime.now(CST).strftime("%Y-%m-%d")
    since = (datetime.now(timezone.utc) - timedelta(hours=25)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    # --- Active PRs in last 24 h ---
    mm_kw_prs = search(
        f"repo:{UPSTREAM} is:pr is:open multimodal updated:>{since}",
        per_page=15,
    )
    mm_label_prs = search(
        f"repo:{UPSTREAM} is:pr is:open label:multi-modal updated:>{since}",
        per_page=15,
    )
    vlm_prs = search(
        f"repo:{UPSTREAM} is:pr is:open VLM updated:>{since}",
        per_page=10,
    )
    vision_prs = search(
        f"repo:{UPSTREAM} is:pr is:open vision-language updated:>{since}",
        per_page=10,
    )

    # --- Contribution targets ---
    mm_no_owner = search(
        f"repo:{UPSTREAM} is:issue is:open label:multi-modal no:assignee",
        per_page=15,
    )
    mm_gfi = search(
        f'repo:{UPSTREAM} is:issue is:open label:multi-modal label:"good first issue"',
        per_page=10,
    )
    mm_help = search(
        f'repo:{UPSTREAM} is:issue is:open label:multi-modal label:"help wanted"',
        per_page=10,
    )

    body = f"""# 🎬 vLLM 多模态每日速报 · {today}

> 每天 **08:00 CST** 自动生成 | fork 已同步上游 [`vllm-project/vllm`](https://github.com/vllm-project/vllm) `main`

---

## 🔥 近24小时 · `multimodal` 关键词 PR

{fmt(mm_kw_prs)}

## 🏷️ 近24小时 · `multi-modal` 标签 PR

{fmt(mm_label_prs)}

## 🤖 近24小时 · VLM 相关 PR

{fmt(vlm_prs)}

## 🖼️ 近24小时 · vision-language 相关 PR

{fmt(vision_prs)}

---

## 🎯 贡献机会 · 多模态 Issues（无人认领）

{fmt(mm_no_owner)}

## ⭐ Good First Issues · 多模态

{fmt(mm_gfi)}

## 🤝 Help Wanted · 多模态

{fmt(mm_help)}

---

<details>
<summary>关于本报告</summary>

- 每天 08:00 CST 由 GitHub Actions 自动生成
- 数据来自 [vllm-project/vllm](https://github.com/vllm-project/vllm) GitHub Search API
- 同步策略：`main` 分支强制对齐上游（`reset --hard upstream/main`）
- [上游最新提交记录](https://github.com/vllm-project/vllm/commits/main)

</details>
"""

    title = f"[Daily Digest] vLLM 多模态速报 {today}"
    url = create_issue(title, body)
    print(f"Issue created: {url}")


if __name__ == "__main__":
    main()
