# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for OmniOpenAIServingChat modalities filtering.

Tests that modalities parameter correctly filters outputs in non-streaming
chat completions while preserving usage/billing information.
"""

from unittest.mock import MagicMock

import pytest
from vllm.entrypoints.openai.protocol import UsageInfo


@pytest.fixture
def mock_request():
    """Create a mock chat completion request."""
    request = MagicMock()
    request.return_token_ids = False
    return request


@pytest.fixture
def mock_omni_output_text():
    """Create a mock text output."""
    output = MagicMock()
    output.final_output_type = "text"
    output.request_output = MagicMock()
    output.request_output.finished = True
    output.request_output.prompt_token_ids = [1] * 10  # 10 prompt tokens
    output.request_output.encoder_prompt_token_ids = None
    output.request_output.num_cached_tokens = 0
    output.request_output.prompt_logprobs = None
    output.request_output.kv_transfer_params = None

    # Mock a single output with token_ids
    mock_output = MagicMock()
    mock_output.token_ids = [1] * 20  # 20 completion tokens
    output.request_output.outputs = [mock_output]

    return output


@pytest.fixture
def mock_omni_output_audio():
    """Create a mock audio output."""
    output = MagicMock()
    output.final_output_type = "audio"
    output.request_output = MagicMock()
    output.request_output.finished = True
    output.request_output.multimodal_output = {"audio": MagicMock()}

    # Mock outputs for audio choice
    mock_output = MagicMock()
    mock_output.index = 0
    mock_output.token_ids = []
    mock_output.stop_reason = None
    output.request_output.outputs = [mock_output]

    return output


@pytest.fixture
def serving_chat():
    """Create OmniOpenAIServingChat instance with mocked dependencies."""
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    instance = object.__new__(OmniOpenAIServingChat)
    instance.enable_prompt_tokens_details = False
    return instance


class TestModalitiesFiltering:
    """Tests for modalities filtering in chat_completion_full_generator."""

    @pytest.mark.asyncio
    async def test_text_only_modality_filters_audio(
        self, serving_chat, mock_request, mock_omni_output_text, mock_omni_output_audio
    ):
        """Test that modalities=["text"] filters out audio outputs."""
        mock_request.modalities = ["text"]

        # Create a mock result generator that yields both text and audio
        async def mock_generator():
            yield mock_omni_output_text
            yield mock_omni_output_audio

        # Mock the methods we need
        serving_chat.get_chat_request_role = MagicMock(return_value="assistant")
        serving_chat._create_text_choice = MagicMock(
            return_value=(
                [MagicMock()],  # choices_data
                UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30),
                None,  # prompt_logprobs
                None,  # prompt_token_ids
                None,  # kv_transfer_params
            )
        )
        serving_chat._create_audio_choice = MagicMock(return_value=[MagicMock()])

        # Call the method
        result = await serving_chat.chat_completion_full_generator(
            request=mock_request,
            result_generator=mock_generator(),
            request_id="test-123",
            model_name="test-model",
            conversation=[],
            tokenizer=MagicMock(),
            request_metadata=MagicMock(),
        )

        # Verify text choice was created
        assert serving_chat._create_text_choice.called
        # Verify audio choice was NOT created (filtered out)
        assert not serving_chat._create_audio_choice.called

        # Verify response has only 1 choice
        assert len(result.choices) == 1
        # Verify usage is populated
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20

    @pytest.mark.asyncio
    async def test_audio_only_modality_filters_text(
        self, serving_chat, mock_request, mock_omni_output_text, mock_omni_output_audio
    ):
        """Test that modalities=["audio"] filters out text but preserves usage."""
        mock_request.modalities = ["audio"]

        # Create a mock result generator that yields both text and audio
        async def mock_generator():
            yield mock_omni_output_text
            yield mock_omni_output_audio

        # Mock the methods we need
        serving_chat.get_chat_request_role = MagicMock(return_value="assistant")
        serving_chat._create_text_choice = MagicMock(
            return_value=(
                [MagicMock()],  # choices_data
                UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30),
                None,  # prompt_logprobs
                None,  # prompt_token_ids
                None,  # kv_transfer_params
            )
        )
        serving_chat._create_audio_choice = MagicMock(return_value=[MagicMock()])

        # Call the method
        result = await serving_chat.chat_completion_full_generator(
            request=mock_request,
            result_generator=mock_generator(),
            request_id="test-123",
            model_name="test-model",
            conversation=[],
            tokenizer=MagicMock(),
            request_metadata=MagicMock(),
        )

        # Verify text choice was NOT created (filtered out)
        assert not serving_chat._create_text_choice.called
        # Verify audio choice was created
        assert serving_chat._create_audio_choice.called

        # Verify response has only 1 choice (audio)
        assert len(result.choices) == 1

        # CRITICAL: Verify usage is still populated even though text was filtered
        # This is the regression that was fixed
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20
        assert result.usage.total_tokens == 30

    @pytest.mark.asyncio
    async def test_multiple_modalities_includes_all(
        self, serving_chat, mock_request, mock_omni_output_text, mock_omni_output_audio
    ):
        """Test that modalities=["text", "audio"] includes both outputs."""
        mock_request.modalities = ["text", "audio"]

        # Create a mock result generator that yields both text and audio
        async def mock_generator():
            yield mock_omni_output_text
            yield mock_omni_output_audio

        # Mock the methods we need
        serving_chat.get_chat_request_role = MagicMock(return_value="assistant")
        serving_chat._create_text_choice = MagicMock(
            return_value=(
                [MagicMock()],  # choices_data
                UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30),
                None,  # prompt_logprobs
                None,  # prompt_token_ids
                None,  # kv_transfer_params
            )
        )
        serving_chat._create_audio_choice = MagicMock(return_value=[MagicMock()])

        # Call the method
        result = await serving_chat.chat_completion_full_generator(
            request=mock_request,
            result_generator=mock_generator(),
            request_id="test-123",
            model_name="test-model",
            conversation=[],
            tokenizer=MagicMock(),
            request_metadata=MagicMock(),
        )

        # Verify both choices were created
        assert serving_chat._create_text_choice.called
        assert serving_chat._create_audio_choice.called

        # Verify response has 2 choices
        assert len(result.choices) == 2
        # Verify usage is populated
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20

    @pytest.mark.asyncio
    async def test_no_modalities_includes_all(
        self, serving_chat, mock_request, mock_omni_output_text, mock_omni_output_audio
    ):
        """Test that when modalities is None, all outputs are included."""
        mock_request.modalities = None

        # Create a mock result generator that yields both text and audio
        async def mock_generator():
            yield mock_omni_output_text
            yield mock_omni_output_audio

        # Mock the methods we need
        serving_chat.get_chat_request_role = MagicMock(return_value="assistant")
        serving_chat._create_text_choice = MagicMock(
            return_value=(
                [MagicMock()],  # choices_data
                UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30),
                None,  # prompt_logprobs
                None,  # prompt_token_ids
                None,  # kv_transfer_params
            )
        )
        serving_chat._create_audio_choice = MagicMock(return_value=[MagicMock()])

        # Call the method
        result = await serving_chat.chat_completion_full_generator(
            request=mock_request,
            result_generator=mock_generator(),
            request_id="test-123",
            model_name="test-model",
            conversation=[],
            tokenizer=MagicMock(),
            request_metadata=MagicMock(),
        )

        # Verify both choices were created
        assert serving_chat._create_text_choice.called
        assert serving_chat._create_audio_choice.called

        # Verify response has 2 choices
        assert len(result.choices) == 2
