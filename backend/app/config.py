"""설정 로딩.

- OpenAI(GPT) 키: 환경변수 GPT_API 에서만 읽는다 (파일에 두지 않는다).
- 국가법령정보센터 OC: 환경변수 LAW_API_OC, 없으면 프로젝트 루트 api_keys.txt 의 `law:` 줄.

      law:<국가법령정보센터 OC>

요청마다 다시 읽으므로 api_keys.txt 를 고친 뒤 서버를 재시작할 필요가 없다.
환경변수는 서버를 띄운 터미널의 값이 쓰이므로, 바꾸면 서버를 다시 띄워야 한다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))  # 시험용 다른 문서 폴더를 쓸 때만 바꾼다
KEY_FILE = Path(os.getenv("API_KEYS_FILE", ROOT_DIR / "api_keys.txt"))


def load_key_file(path: Path = KEY_FILE) -> dict[str, str]:
    """`이름:값` 줄을 읽어 dict 로 반환한다. 값에 ':' 가 있어도 첫 ':' 기준으로만 나눈다."""
    keys: dict[str, str] = {}
    if not path.exists():
        return keys
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        name, sep, value = line.partition(":")
        if sep and name.strip() and value.strip():
            keys[name.strip().lower()] = value.strip()
    return keys


def _usable(value: str | None) -> str | None:
    """'제공예정' 같은 자리표시자(비ASCII 문자 포함)는 키가 없는 것으로 본다."""
    if not value or not value.isascii():
        return None
    return value


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    law_api_oc: str | None
    model: str
    web_search_model: str
    reasoning_effort: str | None  # "low" | "medium" | "high" | None(추론 옵션 미사용)
    max_tool_calls: int
    data_dir: Path = DATA_DIR

    @property
    def has_openai(self) -> bool:
        return self.openai_api_key is not None

    @property
    def has_law_api(self) -> bool:
        return self.law_api_oc is not None


def get_settings() -> Settings:
    keys = load_key_file()
    model = os.getenv("OPENAI_MODEL", "gpt-6-luna")
    effort = os.getenv("REASONING_EFFORT", "low").strip() or None
    return Settings(
        openai_api_key=_usable(os.getenv("GPT_API", "").strip()),
        law_api_oc=_usable(os.getenv("LAW_API_OC") or keys.get("law")),
        model=model,
        web_search_model=os.getenv("WEB_SEARCH_MODEL", model),
        reasoning_effort=effort,
        max_tool_calls=int(os.getenv("MAX_TOOL_CALLS", "15")),
    )
