# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for structured stage configuration builder."""

import json
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from vllm_omni.entrypoints.stage_config import (
    StageConfig,
    StageConfigFactory,
    StageRuntimeConfig,
)


class TestStageRuntimeConfig:
    """Tests for StageRuntimeConfig dataclass."""
    
    def test_runtime_config_creation(self):
        """Test basic runtime config creation."""
        config = StageRuntimeConfig(
            process=True,
            devices="0,1,2",
            max_batch_size=4
        )
        
        assert config.process is True
        assert config.devices == "0,1,2"
        assert config.max_batch_size == 4


class TestStageConfig:
    """Tests for StageConfig dataclass."""
    
    def test_stage_config_creation(self):
        """Test basic stage config creation."""
        runtime = StageRuntimeConfig(
            process=True,
            devices="0",
            max_batch_size=1
        )
        
        config = StageConfig(
            stage_id=0,
            stage_type="diffusion",
            runtime=runtime,
            engine_args={"model": "test-model"},
            final_output=True,
            final_output_type="image"
        )
        
        assert config.stage_id == 0
        assert config.stage_type == "diffusion"
        assert config.runtime.devices == "0"
        assert config.engine_args["model"] == "test-model"
        assert config.final_output is True
        assert config.final_output_type == "image"
    
    def test_to_omegaconf(self):
        """Test conversion to OmegaConf DictConfig."""
        runtime = StageRuntimeConfig(
            process=True,
            devices="0,1",
            max_batch_size=2
        )
        
        config = StageConfig(
            stage_id=0,
            stage_type="diffusion",
            runtime=runtime,
            engine_args={"cache_backend": "cache_dit"},
            final_output=True,
            final_output_type="image"
        )
        
        omega_config = config.to_omegaconf()
        
        assert omega_config.stage_id == 0
        assert omega_config.stage_type == "diffusion"
        assert omega_config.runtime.process is True
        assert omega_config.runtime.devices == "0,1"
        assert omega_config.runtime.max_batch_size == 2
        assert omega_config.engine_args.cache_backend == "cache_dit"
        assert omega_config.final_output is True
        assert omega_config.final_output_type == "image"


class TestDeviceStringGeneration:
    """Tests for device string generation helper."""
    
    def test_device_string_from_parallel_config(self):
        """Test device string generation from parallel_config."""
        # Mock parallel_config with world_size
        parallel_config = Mock()
        parallel_config.world_size = 4
        
        devices = StageConfigFactory._get_device_string(parallel_config)
        
        assert devices == "0,1,2,3"
    
    def test_device_string_from_num_devices(self):
        """Test device string generation from num_devices parameter."""
        devices = StageConfigFactory._get_device_string(None, num_devices=3)
        
        assert devices == "0,1,2"
    
    def test_device_string_default_single_device(self):
        """Test default device string is single device."""
        devices = StageConfigFactory._get_device_string(None, None)
        
        assert devices == "0"
    
    def test_device_string_single_device(self):
        """Test device string with single device."""
        parallel_config = Mock()
        parallel_config.world_size = 1
        
        devices = StageConfigFactory._get_device_string(parallel_config)
        
        assert devices == "0"


class TestCacheConfigNormalization:
    """Tests for cache configuration normalization."""
    
    def test_cache_dit_defaults(self):
        """Test default cache_dit configuration."""
        config = StageConfigFactory._get_default_cache_config("cache_dit")
        
        assert config is not None
        assert config["Fn_compute_blocks"] == 1
        assert config["Bn_compute_blocks"] == 0
        assert config["max_warmup_steps"] == 4
        assert config["residual_diff_threshold"] == 0.24
        assert config["max_continuous_cached_steps"] == 3
        assert config["enable_taylorseer"] is False
        assert config["taylorseer_order"] == 1
        assert config["scm_steps_mask_policy"] is None
        assert config["scm_steps_policy"] == "dynamic"
    
    def test_tea_cache_defaults(self):
        """Test default tea_cache configuration."""
        config = StageConfigFactory._get_default_cache_config("tea_cache")
        
        assert config is not None
        assert config["rel_l1_thresh"] == 0.2
    
    def test_no_cache_defaults(self):
        """Test no defaults for unknown cache backend."""
        config = StageConfigFactory._get_default_cache_config("unknown")
        
        assert config is None
    
    def test_normalize_json_string_cache_config(self):
        """Test normalization of JSON string cache_config."""
        cache_config_str = '{"Fn_compute_blocks": 2, "max_warmup_steps": 6}'
        
        normalized = StageConfigFactory._normalize_cache_config(
            "cache_dit", 
            cache_config_str
        )
        
        assert normalized is not None
        assert normalized["Fn_compute_blocks"] == 2
        assert normalized["max_warmup_steps"] == 6
    
    def test_normalize_invalid_json_uses_defaults(self):
        """Test that invalid JSON falls back to defaults."""
        cache_config_str = '{invalid json}'
        
        normalized = StageConfigFactory._normalize_cache_config(
            "cache_dit", 
            cache_config_str
        )
        
        # Should fall back to defaults for cache_dit
        assert normalized is not None
        assert normalized["Fn_compute_blocks"] == 1
    
    def test_normalize_dict_cache_config(self):
        """Test normalization of dict cache_config."""
        cache_config_dict = {"Fn_compute_blocks": 3}
        
        normalized = StageConfigFactory._normalize_cache_config(
            "cache_dit", 
            cache_config_dict
        )
        
        assert normalized == cache_config_dict
    
    def test_normalize_none_with_backend_uses_defaults(self):
        """Test that None cache_config with backend uses defaults."""
        normalized = StageConfigFactory._normalize_cache_config(
            "cache_dit", 
            None
        )
        
        assert normalized is not None
        assert normalized["Fn_compute_blocks"] == 1
    
    def test_normalize_none_without_backend_returns_none(self):
        """Test that None cache_config without backend returns None."""
        normalized = StageConfigFactory._normalize_cache_config(
            "none", 
            None
        )
        
        assert normalized is None


class TestDefaultDiffusionConfigBuilder:
    """Tests for default diffusion stage configuration builder."""
    
    def test_default_diffusion_config_structure(self):
        """Test that the builder produces correct schema structure."""
        kwargs = {
            "model": "test-model",
            "dtype": "float16",
        }
        
        configs = StageConfigFactory.create_default_diffusion(kwargs)
        
        assert len(configs) == 1
        config = configs[0]
        
        # Verify structure
        assert config.stage_id == 0
        assert config.stage_type == "diffusion"
        assert config.runtime.process is True
        assert config.runtime.devices == "0"
        assert config.runtime.max_batch_size == 1
        assert config.final_output is True
        assert config.final_output_type == "image"
        
        # Verify engine_args
        assert config.engine_args.model == "test-model"
        assert config.engine_args.dtype == "float16"  # Should be stringified
        assert config.engine_args.model_stage == "diffusion"
        assert config.engine_args.cache_backend == "none"
    
    def test_default_diffusion_with_parallel_config(self):
        """Test builder with parallel configuration."""
        parallel_config = Mock()
        parallel_config.world_size = 4
        
        kwargs = {
            "model": "test-model",
            "parallel_config": parallel_config,
        }
        
        configs = StageConfigFactory.create_default_diffusion(kwargs)
        config = configs[0]
        
        # Should generate device string for 4 devices
        assert config.runtime.devices == "0,1,2,3"
    
    def test_default_diffusion_with_cache_backend(self):
        """Test builder with cache backend."""
        kwargs = {
            "model": "test-model",
            "cache_backend": "cache_dit",
        }
        
        configs = StageConfigFactory.create_default_diffusion(kwargs)
        config = configs[0]
        
        # Should have cache_dit defaults
        assert config.engine_args.cache_backend == "cache_dit"
        assert config.engine_args.cache_config is not None
        assert config.engine_args.cache_config["Fn_compute_blocks"] == 1
    
    def test_default_diffusion_preserves_kwargs(self):
        """Test that builder preserves additional kwargs in engine_args."""
        kwargs = {
            "model": "test-model",
            "custom_param": "custom_value",
            "another_param": 42,
        }
        
        configs = StageConfigFactory.create_default_diffusion(kwargs)
        config = configs[0]
        
        # Should preserve custom parameters
        assert config.engine_args.custom_param == "custom_value"
        assert config.engine_args.another_param == 42
    
    def test_dtype_stringification(self):
        """Test that dtype is converted to string."""
        # Use a mock object for dtype
        class DType:
            def __str__(self):
                return "torch.float16"
        
        kwargs = {
            "model": "test-model",
            "dtype": DType(),
        }
        
        configs = StageConfigFactory.create_default_diffusion(kwargs)
        config = configs[0]
        
        assert config.engine_args.dtype == "torch.float16"


class TestAsyncDiffusionConfigBuilder:
    """Tests for async diffusion stage configuration builder."""
    
    def test_async_diffusion_with_parallel_config(self):
        """Test async builder with parallel_config."""
        parallel_config = Mock()
        parallel_config.world_size = 2
        
        kwargs = {
            "parallel_config": parallel_config,
            "vae_use_slicing": True,
            "vae_use_tiling": False,
        }
        
        configs = StageConfigFactory.create_default_diffusion_async(kwargs)
        config = configs[0]
        
        assert config.runtime.devices == "0,1"
        assert config.engine_args.parallel_config == parallel_config
        assert config.engine_args.vae_use_slicing is True
        assert config.engine_args.vae_use_tiling is False
    
    def test_async_diffusion_builds_parallel_config(self):
        """Test async builder builds parallel_config from parameters."""
        kwargs = {
            "tensor_parallel_size": 2,
            "ulysses_degree": 2,
            "ring_degree": 1,
        }
        
        configs = StageConfigFactory.create_default_diffusion_async(kwargs)
        config = configs[0]
        
        # Should build parallel_config and device string
        # tensor_parallel_size=2 * sequence_parallel_size=2 = 4 devices
        assert config.runtime.devices == "0,1,2,3"
        assert config.engine_args.parallel_config is not None
        assert config.engine_args.parallel_config.tensor_parallel_size == 2
        assert config.engine_args.parallel_config.ulysses_degree == 2
    
    def test_async_diffusion_includes_async_specific_args(self):
        """Test that async builder includes async-specific engine args."""
        kwargs = {
            "enable_cpu_offload": True,
            "enforce_eager": True,
        }
        
        configs = StageConfigFactory.create_default_diffusion_async(kwargs)
        config = configs[0]
        
        assert config.engine_args.enable_cpu_offload is True
        assert config.engine_args.enforce_eager is True
        assert config.engine_args.model_stage == "diffusion"


class TestCacheConfigIntegration:
    """Integration tests for cache config normalization."""
    
    def test_cache_backend_integration(self):
        """Test cache_backend arguments are correctly nested in config."""
        kwargs = {
            "model": "test-model",
            "cache_backend": "cache_dit",
        }
        
        configs = StageConfigFactory.create_default_diffusion(kwargs)
        config = configs[0]
        
        # Verify cache_dit defaults are in engine_args
        assert config.engine_args.cache_backend == "cache_dit"
        assert config.engine_args.cache_config is not None
        assert "Fn_compute_blocks" in config.engine_args.cache_config
        assert "max_warmup_steps" in config.engine_args.cache_config
    
    def test_custom_cache_config_override(self):
        """Test custom cache_config overrides defaults."""
        custom_config = {"Fn_compute_blocks": 5, "custom_key": "custom_value"}
        
        kwargs = {
            "model": "test-model",
            "cache_backend": "cache_dit",
            "cache_config": custom_config,
        }
        
        configs = StageConfigFactory.create_default_diffusion(kwargs)
        config = configs[0]
        
        # Should use custom config
        assert config.engine_args.cache_config["Fn_compute_blocks"] == 5
        assert config.engine_args.cache_config["custom_key"] == "custom_value"
