"""Tests for /restore gateway slash command.

/restore should import a tail-focused summary from the previous session into the
current session without switching threads like /resume does.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/restore", platform=Platform.TELEGRAM,
                user_id="12345", chat_id="67890"):
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner(tmp_path):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._session_db = None
    runner._running_agents = {}
    runner.config = SimpleNamespace(sessions_dir=tmp_path)

    mock_session_entry = MagicMock()
    mock_session_entry.session_id = "current_session"
    mock_session_entry.session_key = "telegram:12345:67890"
    mock_store = MagicMock()
    mock_store.get_or_create_session.return_value = mock_session_entry
    runner.session_store = mock_store
    return runner


class TestRestoreSummaryHelpers:
    def test_restore_summary_is_tail_focused_and_surfaces_unfinished_work(self):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        record = {
            "session_id": "prev_123",
            "messages": [
                {"role": "user", "content": "six hours ago old context that should not dominate restore"},
                {"role": "assistant", "content": "old answer"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "call_id": "todo_call_1",
                            "type": "function",
                            "function": {"name": "todo", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "todo_call_1",
                    "content": json.dumps(
                        {
                            "todos": [
                                {"id": "rerun", "content": "Rerun targeted tests", "status": "in_progress"},
                                {"id": "done", "content": "Backup repo", "status": "completed"},
                            ]
                        }
                    ),
                },
                {"role": "user", "content": "continue the update protocol"},
                {"role": "assistant", "content": "We updated successfully; next step is rerunning targeted tests."},
            ],
        }

        restore = runner._build_restore_summary_from_session_record(record)
        summary = restore["summary"]

        assert "continue the update protocol" in summary
        assert "Rerun targeted tests" in summary
        assert "six hours ago old context" not in summary
        assert restore["unfinished_todos"] == ["Rerun targeted tests"]


class TestHandleRestoreCommand:
    @pytest.mark.asyncio
    async def test_restore_uses_previous_session_and_appends_summary_to_current_session(self, tmp_path):
        runner = _make_runner(tmp_path)

        previous = {
            "session_id": "previous_session",
            "session_start": "2026-03-12T03:00:00",
            "messages": [
                {"role": "user", "content": "finish the CAAM fix"},
                {"role": "assistant", "content": "The work is done; just report status."},
            ],
        }
        current = {
            "session_id": "current_session",
            "session_start": "2026-03-12T04:00:00",
            "messages": [
                {"role": "user", "content": "hello"},
            ],
        }

        (tmp_path / "session_previous_session.json").write_text(json.dumps(previous), encoding="utf-8")
        (tmp_path / "session_current_session.json").write_text(json.dumps(current), encoding="utf-8")

        event = _make_event(text="/restore")
        result = await runner._handle_restore_command(event)

        assert "Context restored from previous session" in result
        runner.session_store.append_to_transcript.assert_called_once()
        call_args = runner.session_store.append_to_transcript.call_args[0]
        assert call_args[0] == "current_session"
        assert call_args[1]["role"] == "assistant"
        assert "finish the CAAM fix" in call_args[1]["content"]

    @pytest.mark.asyncio
    async def test_restore_auto_continues_when_unfinished_work_exists(self, tmp_path):
        runner = _make_runner(tmp_path)
        runner._handle_message = AsyncMock(return_value="continued from restore")

        previous = {
            "session_id": "previous_session",
            "session_start": "2026-03-12T03:00:00",
            "messages": [
                {"role": "user", "content": "finish the CAAM fix"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "call_id": "todo_call_2",
                            "type": "function",
                            "function": {"name": "todo", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "todo_call_2",
                    "content": json.dumps(
                        {
                            "todos": [
                                {"id": "tests", "content": "Run targeted tests", "status": "pending"},
                            ]
                        }
                    ),
                },
                {"role": "assistant", "content": "The next step is to run targeted tests and report the result."},
            ],
        }
        current = {
            "session_id": "current_session",
            "session_start": "2026-03-12T04:00:00",
            "messages": [{"role": "user", "content": "hello"}],
        }

        (tmp_path / "session_previous_session.json").write_text(json.dumps(previous), encoding="utf-8")
        (tmp_path / "session_current_session.json").write_text(json.dumps(current), encoding="utf-8")

        event = _make_event(text="/restore")
        result = await runner._handle_restore_command(event)

        assert result == "continued from restore"
        runner._handle_message.assert_awaited_once()
        continued_event = runner._handle_message.await_args.args[0]
        assert "Continue from the restored tail context" in continued_event.text
        assert continued_event.source == event.source

    @pytest.mark.asyncio
    async def test_restore_returns_error_when_no_previous_session_exists(self, tmp_path):
        runner = _make_runner(tmp_path)
        current = {
            "session_id": "current_session",
            "session_start": "2026-03-12T04:00:00",
            "messages": [{"role": "user", "content": "hello"}],
        }
        (tmp_path / "session_current_session.json").write_text(json.dumps(current), encoding="utf-8")

        event = _make_event(text="/restore")
        result = await runner._handle_restore_command(event)

        assert "No previous session record found" in result
        runner.session_store.append_to_transcript.assert_not_called()


class TestRestoreInHelp:
    @pytest.mark.asyncio
    async def test_restore_in_help_output(self, tmp_path):
        runner = _make_runner(tmp_path)
        from gateway.hooks import HookRegistry
        runner.hooks = HookRegistry()
        event = _make_event(text="/help")

        result = await runner._handle_help_command(event)
        assert "/restore" in result

    def test_restore_is_known_command(self):
        from gateway.run import GatewayRunner
        import inspect

        source = inspect.getsource(GatewayRunner._handle_message)
        assert '"restore"' in source
