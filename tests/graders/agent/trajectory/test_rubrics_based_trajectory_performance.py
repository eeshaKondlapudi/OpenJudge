#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for RubricsGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using RubricsGrader as an example:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real API)

Example:
    Run all tests:
    ```bash
    pytest tests/graders/agent/trajectory/test_rubrics_based_trajectory_performance.py -v
    ```

    Run only unit tests:
    ```bash
    pytest tests/graders/agent/trajectory/test_rubrics_based_trajectory_performance.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    pytest tests/graders/agent/trajectory/test_rubrics_based_trajectory_performance.py -m quality
    ```
"""

import json
import os
from unittest.mock import AsyncMock

import pytest

from rm_gallery.core.graders.agent.trajectory.rubrics_based_trajectory_performance import (
    RubricsBasedTrajectoryPerformance,
)
from rm_gallery.core.graders.schema import GraderError
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.models.schema.prompt_template import LanguageEnum

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestRubricsBasedTrajectoryPerformanceUnit:
    """Unit tests for RubricsBasedTrajectoryPerformance - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization with different languages"""
        mock_model = AsyncMock()

        # Test Chinese initialization
        grader_zh = RubricsBasedTrajectoryPerformance(
            model=mock_model,
            language=LanguageEnum.ZH,
        )
        assert grader_zh.name == "rubrics_based_trajectory_performance"
        assert grader_zh.model == mock_model
        assert grader_zh.language == LanguageEnum.ZH

        # Test English initialization
        grader_en = RubricsBasedTrajectoryPerformance(
            model=mock_model,
            language=LanguageEnum.EN,
        )
        assert grader_en.name == "rubrics_based_trajectory_performance"
        assert grader_en.language == LanguageEnum.EN

    @pytest.mark.asyncio
    async def test_successful_evaluation_partial_passed(self):
        """Test successful evaluation with some check points failed"""
        # Create grader first
        mock_model = AsyncMock()
        grader = RubricsBasedTrajectoryPerformance(model=mock_model)

        # Setup rubrics
        rubrics = [
            {
                "dimension": "完整性",
                "description": "报告是否完整",
                "check_points": ["覆盖所有关键指标", "包含时间信息", "包含来源信息"],
            },
            {
                "dimension": "准确性",
                "description": "数据是否准确",
                "check_points": ["数值正确", "来源可靠"],
            },
        ]

        # Setup mock response with some failures
        dimension_evaluations = [
            {
                "dimension": "完整性",
                "reasoning": "报告部分完整，缺少来源信息",
                "check_points_passed": ["覆盖所有关键指标", "包含时间信息"],
                "check_points_failed": ["包含来源信息"],
            },
            {
                "dimension": "准确性",
                "reasoning": "数据准确但来源不够可靠",
                "check_points_passed": ["数值正确"],
                "check_points_failed": ["来源可靠"],
            },
        ]

        mock_callback_response = AsyncMock()
        mock_callback_response.parsed = {
            "dimension_evaluations": dimension_evaluations,
        }

        # Create callback
        callback = grader._create_rubrics_callback(language=LanguageEnum.ZH)
        callback_result = callback(mock_callback_response)

        # Setup mock response
        mock_response = AsyncMock()
        mock_response.parsed = callback_result

        mock_model.achat = AsyncMock(return_value=mock_response)

        # Execute test
        messages = [
            {"role": "user", "content": "分析公司财报"},
            {"role": "assistant", "content": "财务分析报告..."},
        ]

        result = await grader.aevaluate(messages=messages, rubrics=rubrics)

        # Assertions
        assert result is not None
        assert result.score < 1.0  # Not all check points passed
        # 完整性: 2/3=0.667, 准确性: 1/2=0.5
        # 平均总分: (0.667 + 0.5) / 2 = 0.583
        assert abs(result.score - 0.583) < 0.02

    @pytest.mark.asyncio
    async def test_invalid_rubrics_edge_cases(self):
        """Test edge cases with invalid or empty rubrics"""
        mock_model = AsyncMock()
        grader = RubricsBasedTrajectoryPerformance(model=mock_model)

        messages = [
            {"role": "user", "content": "测试问题"},
            {"role": "assistant", "content": "测试回答"},
        ]

        # Test with empty rubrics list
        result = await grader.aevaluate(messages=messages, rubrics=[])
        assert isinstance(result, GraderError)
        assert result.error == "Invalid or empty rubrics, must be a list"

        # Test with None rubrics
        result = await grader.aevaluate(messages=messages, rubrics=None)
        assert isinstance(result, GraderError)
        assert result.error == "Invalid or empty rubrics, must be a list"

        # Model should not be called for invalid rubrics
        mock_model.achat.assert_not_called()

    @pytest.mark.asyncio
    async def test_incomplete_messages_edge_cases(self):
        """Test edge cases with incomplete or invalid messages"""
        mock_model = AsyncMock()
        grader = RubricsBasedTrajectoryPerformance(model=mock_model)

        rubrics = [
            {
                "dimension": "完整性",
                "description": "测试",
                "check_points": ["检查点1"],
            }
        ]

        # Test with empty messages
        result = await grader.aevaluate(messages=[], rubrics=rubrics)
        assert isinstance(result, GraderError)
        assert result.error == "Empty query or tool_calls or final_response"

        # Test with messages without assistant response
        messages = [
            {"role": "user", "content": "测试问题"},
            {"role": "tool", "content": "工具返回..."},
        ]
        result = await grader.aevaluate(messages=messages, rubrics=rubrics)
        assert isinstance(result, GraderError)
        assert result.error == "Empty query or tool_calls or final_response"

        # Model should not be called for incomplete input
        mock_model.achat.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling when LLM call fails"""
        # Setup mock to raise exception
        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(side_effect=Exception("API Error: Connection timeout"))

        # Create grader
        grader = RubricsBasedTrajectoryPerformance(model=mock_model)

        rubrics = [
            {
                "dimension": "完整性",
                "description": "测试",
                "check_points": ["检查点1"],
            }
        ]

        # Execute test
        messages = [
            {"role": "user", "content": "测试问题"},
            {"role": "assistant", "content": "测试回答"},
        ]

        result = await grader.aevaluate(messages=messages, rubrics=rubrics)

        # Assertions
        assert isinstance(result, GraderError)
        assert result.name == "rubrics_based_trajectory_performance"
        assert "API Error" in result.error or "Connection timeout" in result.error

    def test_extraction_format_with_nested_messages(self):
        """Test extraction handles nested message format correctly"""
        mock_model = AsyncMock()
        grader = RubricsBasedTrajectoryPerformance(model=mock_model)

        # Test with nested message structure
        messages = [
            {"message": {"role": "system", "content": "System prompt"}},
            {"message": {"role": "user", "content": "测试问题"}},
            {"message": {"role": "assistant", "content": "测试回答"}},
        ]

        # Extract components
        query, tool_calls, final_response = grader._extract_info_from_messages(messages)

        # Assertions
        assert query == "测试问题"
        assert final_response == "测试回答"

    def test_extraction_with_tool_calls(self):
        """Test extraction handles tool calls correctly"""
        mock_model = AsyncMock()
        grader = RubricsBasedTrajectoryPerformance(model=mock_model)

        messages = [
            {"role": "user", "content": "查询数据"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_data",
                            "arguments": '{"query": "test"}',
                        }
                    }
                ],
            },
            {"role": "tool", "content": "搜索结果数据"},
            {"role": "assistant", "content": "根据搜索结果，分析如下..."},
        ]

        query, tool_calls_str, final_response = grader._extract_info_from_messages(messages)

        # Should extract tool call information
        assert "search_data" in tool_calls_str
        assert "query" in tool_calls_str
        assert final_response == "根据搜索结果，分析如下..."


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)


@pytest.mark.skipif(
    not RUN_QUALITY_TESTS,
    reason="Requires API keys and base URL to run quality tests",
)
@pytest.mark.quality
class TestRubricsBasedTrajectoryPerformanceQuality:
    """Quality tests for RubricsBasedTrajectoryPerformance - testing evaluation quality"""

    @pytest.fixture
    def model(self):
        """Return OpenAIChatModel instance based on environment variables"""
        if OPENAI_API_KEY:
            config = {
                "model": "qwen3-max",
                "api_key": OPENAI_API_KEY,
                "max_tokens": 4096,
            }
            if OPENAI_BASE_URL:
                config["base_url"] = OPENAI_BASE_URL
            return OpenAIChatModel(**config)
        else:
            # This shouldn't happen because tests are skipped if keys aren't configured
            raise RuntimeError("No API key configured")

    @pytest.fixture
    def test_data(self):
        """Load test data from JSON file"""
        test_data_path = os.path.join(os.path.dirname(__file__), "test_data_with_rubrics.json")
        with open(test_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    @pytest.mark.asyncio
    async def test_real_data_performance(self, model, test_data):
        """Test evaluation quality on real research data with rubrics"""
        # Create grader with real model
        grader = RubricsBasedTrajectoryPerformance(model=model, language=LanguageEnum.ZH)

        # Use test data
        messages = test_data["messages"]
        rubrics = test_data["rubrics"]

        # Execute evaluation with real model
        result = await grader.aevaluate(messages=messages, rubrics=rubrics)

        # Assertions
        assert result.name == "rubrics_based_trajectory_performance"
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

        # Verify metadata structure
        assert "dimension_scores" in result.metadata
        assert "dimension_evaluations" in result.metadata
        # Should have evaluated all dimensions
        assert len(result.metadata["dimension_evaluations"]) == len(rubrics)

        print(f"\n=== Real Data Quality Test ===")
        print(f"Score: {result.score:.4f}")
        print(f"Number of Dimensions: {len(result.metadata['dimension_evaluations'])}")
        print(f"Dimension Scores: {result.metadata.get('dimension_scores')}")
        print(f"Reason Preview: {result.reason[:500]}...")

    @pytest.mark.asyncio
    async def test_good_response_performance(self, model):
        """Test that grader correctly identifies good responses"""
        # Create grader with real model
        grader = RubricsBasedTrajectoryPerformance(model=model, language=LanguageEnum.ZH)

        # Create a good response scenario
        messages = [
            {"role": "user", "content": "分析贵州茅台的盈利能力和财务状况"},
            {
                "role": "tool",
                "content": "截至2025年6月30日，贵州茅台净利润454.03亿元，净资产收益率17.89%，毛利率91.30%。",
            },
            {
                "role": "assistant",
                "content": """根据最新财务数据（截至2025年6月30日），贵州茅台展现出卓越的盈利能力：

1. **盈利能力强劲**：上半年实现净利润454.03亿元，同比增长8.89%
2. **资产回报率高**：净资产收益率达17.89%，显示出高效的资本运用
3. **利润率优异**：毛利率高达91.30%，体现了强大的定价权和品牌护城河

综合以上数据，贵州茅台的盈利能力和财务状况均处于行业领先水平。""",
            },
        ]

        rubrics = [
            {
                "dimension": "数据准确性",
                "description": "评估数据引用是否准确完整",
                "check_points": [
                    "数值引用准确",
                    "包含时间信息",
                    "数据来源明确",
                ],
            },
            {
                "dimension": "分析完整性",
                "description": "评估分析是否全面覆盖关键点",
                "check_points": [
                    "覆盖净利润分析",
                    "覆盖ROE分析",
                    "给出综合结论",
                ],
            },
        ]

        result = await grader.aevaluate(messages=messages, rubrics=rubrics)

        # A good response should score high
        print(f"\n=== Good Response Quality Test ===")
        print(f"Score: {result.score:.4f}")
        print(f"Dimension Scores: {result.metadata.get('dimension_scores')}")
        print(f"Reason: {result.reason}")

        # Should have high score (though we allow some variance due to LLM)
        assert result.score >= 0.5

    @pytest.mark.asyncio
    async def test_poor_response_performance(self, model):
        """Test that grader correctly identifies poor responses"""
        # Create grader with real model
        grader = RubricsBasedTrajectoryPerformance(model=model, language=LanguageEnum.ZH)

        # Create a poor response scenario with intentional issues
        messages = [
            {"role": "user", "content": "分析贵州茅台的盈利能力和财务状况"},
            {
                "role": "tool",
                "content": "截至2025年6月30日，贵州茅台净利润454.03亿元，净资产收益率17.89%，毛利率91.30%。",
            },
            {
                "role": "assistant",
                "content": """贵州茅台的财务表现不错。

公司盈利很好，财务状况健康。""",  # 故意简短、缺乏具体数据
            },
        ]

        rubrics = [
            {
                "dimension": "数据准确性",
                "description": "评估数据引用是否准确完整",
                "check_points": [
                    "数值引用准确",
                    "包含时间信息",
                    "数据来源明确",
                ],
            },
            {
                "dimension": "分析完整性",
                "description": "评估分析是否全面覆盖关键点",
                "check_points": [
                    "覆盖净利润分析",
                    "覆盖ROE分析",
                    "给出具体数据支撑",
                ],
            },
        ]

        result = await grader.aevaluate(messages=messages, rubrics=rubrics)

        # A poor response should score low
        print(f"\n=== Poor Response Quality Test ===")
        print(f"Score: {result.score:.4f}")
        print(f"Dimension Scores: {result.metadata.get('dimension_scores')}")
        print(f"Reason: {result.reason}")

        # Should have low score
        assert result.score < 0.5

    @pytest.mark.asyncio
    async def test_consistency_performance(self, model):
        """Test grader evaluation consistency on same input"""
        # Create grader with real model
        grader = RubricsBasedTrajectoryPerformance(model=model, language=LanguageEnum.ZH)

        # Test data
        messages = [
            {"role": "user", "content": "分析财报"},
            {
                "role": "tool",
                "content": "截至2025年6月30日，净利润454.03亿元，ROE为17.89%。",
            },
            {
                "role": "assistant",
                "content": "根据数据显示，截至2025年6月30日，净利润为454.03亿元，ROE达到17.89%，财务表现稳健。",
            },
        ]

        rubrics = [
            {
                "dimension": "数据准确性",
                "description": "数据引用准确",
                "check_points": ["数值正确", "时间明确"],
            }
        ]

        # Run evaluation twice
        result1 = await grader.aevaluate(messages=messages, rubrics=rubrics)
        result2 = await grader.aevaluate(messages=messages, rubrics=rubrics)

        # Assertions
        assert result1.name == result2.name
        assert isinstance(result1.score, (int, float))
        assert isinstance(result2.score, (int, float))

        print(f"\n=== Consistency Quality Test ===")
        print(f"Run 1 Score: {result1.score:.4f}")
        print(f"Run 2 Score: {result2.score:.4f}")
        print(f"Score Difference: {abs(result1.score - result2.score):.4f}")

        # Scores should be similar (allow some variance)
        score_diff = abs(result1.score - result2.score)
        if score_diff > 0.3:
            print(f"WARNING: Large score variance detected: {score_diff}")

    @pytest.mark.asyncio
    async def test_english_language_performance(self, model):
        """Test grader works correctly with English language"""
        # Create grader with English language
        grader = RubricsBasedTrajectoryPerformance(model=model, language=LanguageEnum.EN)

        # English test data
        messages = [
            {"role": "user", "content": "Analyze the company's financial performance"},
            {
                "role": "tool",
                "content": "As of June 30, 2025: Net profit 45.4 billion yuan, ROE 17.89%, Gross margin 91.30%.",
            },
            {
                "role": "assistant",
                "content": "Based on the latest financial data as of June 30, 2025: Net profit reached 45.4 billion yuan, ROE was 17.89%, and gross margin stood at 91.30%, demonstrating strong profitability.",
            },
        ]

        rubrics = [
            {
                "dimension": "Data Accuracy",
                "description": "Evaluate data citation accuracy",
                "check_points": ["Accurate values", "Clear timeframe"],
            },
            {
                "dimension": "Analysis Completeness",
                "description": "Evaluate analysis coverage",
                "check_points": ["Covers key metrics", "Provides conclusion"],
            },
        ]

        result = await grader.aevaluate(messages=messages, rubrics=rubrics)

        # Assertions
        assert result.name == "rubrics_based_trajectory_performance"
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

        print(f"\n=== English Language Quality Test ===")
        print(f"Score: {result.score:.4f}")
        print(f"Dimension Scores: {result.metadata.get('dimension_scores')}")
        print(f"Reason: {result.reason[:300]}...")

        # Should have evaluated all dimensions
        assert len(result.metadata["dimension_evaluations"]) == len(rubrics)
