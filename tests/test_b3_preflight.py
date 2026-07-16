"""scripts/b3_preflight.py 单元测试（PATCH 9 §9.4 B3 驱动加固：HEAD/净树/磁盘三条起跑前断言）。

tmp_git_repo fixture 在 tmp_path 里建一个最小 git 仓库（git init + 一次 commit），user.email/
user.name 走 subprocess 的 env 传入，不依赖宿主机全局 git config，保证测试在任何环境下确定性
可跑。
"""
from __future__ import annotations
import os
import subprocess
import pytest
from scripts.b3_preflight import main, preflight

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@example.com",
}


@pytest.fixture
def tmp_git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {**os.environ, **_GIT_ENV}
    subprocess.run(["git", "init"], cwd=repo, env=env, check=True, capture_output=True)
    (repo / "README.md").write_text("init\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, env=env, check=True,
                   capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, env=env, check=True,
                   capture_output=True)
    return repo


def _head(repo) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def test_preflight_head_mismatch_fails(tmp_git_repo):
    errs = preflight("0000000", min_free_gb=0, repo_root=str(tmp_git_repo))
    assert any("HEAD" in e for e in errs)


def test_preflight_dirty_tree_fails(tmp_git_repo):
    (tmp_git_repo / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")
    errs = preflight(None, min_free_gb=0, repo_root=str(tmp_git_repo))
    assert any("不净" in e for e in errs)


def test_preflight_disk_threshold(tmp_git_repo):
    # 天文数字级别的余量要求：任何真实磁盘都不可能满足，必 FAIL。
    errs_fail = preflight(None, min_free_gb=10**6, repo_root=str(tmp_git_repo))
    assert any("磁盘" in e for e in errs_fail)
    # 零余量要求：该项必然通过（干净树 + HEAD 未指定 → 全绿）。
    errs_pass = preflight(None, min_free_gb=0, repo_root=str(tmp_git_repo))
    assert errs_pass == []


def test_preflight_all_green(tmp_git_repo):
    head7 = _head(tmp_git_repo)[:7]
    errs = preflight(head7, min_free_gb=0, repo_root=str(tmp_git_repo))
    assert errs == []


def test_main_rejects_empty_expected_head():
    """--expected-head "" 空串防退化（终审修复 Fix 4）：str.startswith("") 对任意 HEAD 都
    判定为"匹配"，若放行会让 HEAD 断言在传入空串时静默失效（调用方一旦把可能为空的变量
    原样传进来就等于没检查、且不报错——比"忘了传"更危险，因为看起来像是传了）。argparse
    层直接拒绝显式空串，逼调用方要么不传该参数（显式跳过检查）要么传真实 commit 前缀。
    """
    with pytest.raises(SystemExit) as exc_info:
        main(["--expected-head", "", "--min-free-gb", "0"])
    assert exc_info.value.code != 0
