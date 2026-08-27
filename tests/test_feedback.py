"""Tests for feedback and Slack notification module."""

from unittest.mock import MagicMock, patch
import pandas as pd

from redoxquant.feedback import send_slack_alert
from redoxquant.synthetic import generate_standards


def test_send_slack_alert_mock():
    std = generate_standards(seed=1, cv=0.01)

    # Mock urllib.request.urlopen context manager
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:

        ok = send_slack_alert(
            "https://hooks.slack.com/services/MOCK/TOKEN/XYZ",
            title="Test Run Complete",
            message="Tocilizumab assay run processed with 0 QC errors.",
            df=std,
        )
        assert ok
        assert mock_urlopen.called


def test_submit_feedback_mock():
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        from redoxquant.feedback import submit_feedback
        ok = submit_feedback(
            "Could you add support for 4-channel Octet import format?",
            user_email="scientist@biotech.com",
            user_name="Dr. Alex",
            category="Feature Request",
            webhook_url="https://hooks.slack.com/services/MOCK/TOKEN/XYZ",
        )
        assert ok
        assert mock_urlopen.called


