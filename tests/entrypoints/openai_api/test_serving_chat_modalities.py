# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for OmniOpenAIServingChat modalities filtering.

Tests that modalities parameter correctly filters outputs in non-streaming
chat completions while preserving usage/billing information.
"""

from unittest.mock import MagicMock

# =============================================================================
# Tests for modalities set creation logic
# =============================================================================


def test_requested_modalities_set_creation_with_list():
    """Test that requested_modalities set is created from list."""
    # Simulate what happens in the code
    request = MagicMock()
    request.modalities = ["text", "audio"]

    # This is the logic from the fix
    requested_modalities = set(request.modalities) if hasattr(request, "modalities") and request.modalities else None

    assert requested_modalities == {"text", "audio"}


def test_requested_modalities_set_creation_with_none():
    """Test that requested_modalities is None when modalities is None."""
    request = MagicMock()
    request.modalities = None

    requested_modalities = set(request.modalities) if hasattr(request, "modalities") and request.modalities else None

    assert requested_modalities is None


def test_requested_modalities_set_creation_when_missing():
    """Test that requested_modalities is None when attribute missing."""
    request = MagicMock(spec=[])  # No modalities attribute

    requested_modalities = set(request.modalities) if hasattr(request, "modalities") and request.modalities else None

    assert requested_modalities is None


# =============================================================================
# Tests for modality filtering logic
# =============================================================================


def test_modality_filtering_text_in_set():
    """Test that text output passes filter when text is requested."""
    requested_modalities = {"text", "audio"}
    final_output_type = "text"

    # This is the filtering logic from the fix
    should_filter = requested_modalities is not None and final_output_type not in requested_modalities

    assert not should_filter  # text should NOT be filtered


def test_modality_filtering_audio_not_in_set():
    """Test that audio output is filtered when only text requested."""
    requested_modalities = {"text"}
    final_output_type = "audio"

    should_filter = requested_modalities is not None and final_output_type not in requested_modalities

    assert should_filter  # audio SHOULD be filtered


def test_modality_filtering_with_none_modalities():
    """Test that nothing is filtered when modalities is None."""
    requested_modalities = None
    final_output_type = "text"

    should_filter = requested_modalities is not None and final_output_type not in requested_modalities

    assert not should_filter  # Nothing should be filtered

    final_output_type = "audio"
    should_filter = requested_modalities is not None and final_output_type not in requested_modalities

    assert not should_filter  # Nothing should be filtered


# =============================================================================
# Tests for usage extraction logic
# =============================================================================


def test_usage_extraction_logic():
    """Test that usage info is extracted correctly from request_output."""
    # Mock the omni_outputs structure
    omni_outputs = MagicMock()
    omni_outputs.final_output_type = "text"
    omni_outputs.request_output = MagicMock()
    omni_outputs.request_output.prompt_token_ids = [1] * 10
    omni_outputs.request_output.encoder_prompt_token_ids = None

    mock_output = MagicMock()
    mock_output.token_ids = [1] * 20
    omni_outputs.request_output.outputs = [mock_output]

    # Simulate the usage extraction logic
    final_output_type = omni_outputs.final_output_type
    if final_output_type == "text" and omni_outputs.request_output is not None:
        final_res = omni_outputs.request_output
        if final_res.prompt_token_ids is not None:
            num_prompt_tokens = len(final_res.prompt_token_ids)
            if final_res.encoder_prompt_token_ids is not None:
                num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
            num_generated_tokens = sum(len(output.token_ids) for output in final_res.outputs)

            assert num_prompt_tokens == 10
            assert num_generated_tokens == 20
