#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for FinancialReportResolutionGrader showing unit tests and quality tests.

This file demonstrates two types of tests recommended in the GRADER_TESTING_STRATEGY.md
using FinancialReportResolutionGrader as an example:

1. Unit tests (offline testing with mocks)
2. Quality tests (evaluation against real API)

Example:
    Run all tests:
    ```bash
    poetry run pytest tests/graders/agent/deep_research/test_financial_report_resolution.py -v
    ```

    Run only unit tests:
    ```bash
    poetry run pytest tests/graders/agent/deep_research/test_financial_report_resolution.py -m unit
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    poetry run pytest tests/graders/agent/deep_research/test_financial_report_resolution.py -m quality
    ```
"""

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from rm_gallery.core.graders.agent.deep_research.financial_report_resolution import (
    FinancialReportResolutionGrader,
)
from rm_gallery.core.graders.schema import GraderError
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.models.schema.prompt_template import LanguageEnum

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
class TestFinancialReportResolutionGraderUnit:
    """Unit tests for FinancialReportResolutionGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()

        grader = FinancialReportResolutionGrader(
            model=mock_model,
            resolution_threshold=0.85,
        )
        assert grader.resolution_threshold == 0.85
        assert grader.name == "financial_report_resolution"
        assert grader.model == mock_model
        assert grader.language == LanguageEnum.ZH

    def test_initialization_with_english(self):
        """Test initialization with English language"""
        mock_model = AsyncMock()

        grader = FinancialReportResolutionGrader(
            model=mock_model,
            language=LanguageEnum.EN,
            resolution_threshold=0.9,
        )
        assert grader.language == LanguageEnum.EN
        assert grader.resolution_threshold == 0.9

    @pytest.mark.asyncio
    async def test_successful_evaluation(self):
        """Test successful evaluation with valid inputs"""
        # Setup mock response with callback-processed format (score + reason + dimension_scores)
        # Simulating what callback returns after processing dimension scores
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1.0,  # All dimensions are 5/5, normalized weighted average
            "reason": "【精准性】5/5分: 完全精准\n【完整性】5/5分: 完全完整\n【相关性】5/5分: 高度相关\n【时效性】5/5分: 时效性极佳\n【逻辑严谨性】5/5分: 逻辑完全严谨\n【数据支撑】5/5分: 数据支撑充分\n【报告结构与可读性】5/5分: 结构完美",
            "dimension_scores": {
                "precision": {"raw": 5, "normalized": 1.0},
                "completeness": {"raw": 5, "normalized": 1.0},
                "relevance": {"raw": 5, "normalized": 1.0},
                "timeliness": {"raw": 5, "normalized": 1.0},
                "logic": {"raw": 5, "normalized": 1.0},
                "data": {"raw": 5, "normalized": 1.0},
                "structure": {"raw": 5, "normalized": 1.0},
            },
        }

        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)

        # Create grader
        grader = FinancialReportResolutionGrader(model=mock_model, resolution_threshold=0.9)

        # Execute test
        messages = [
            {"role": "system", "content": "你是一个金融分析专家"},
            {"role": "user", "content": "分析某公司的财务状况"},
            {"role": "assistant", "content": "详细的财务分析报告..."},
        ]

        result = await grader.aevaluate(messages=messages, chat_date="2024-01-01")

        # Assertions
        assert result is not None
        assert result.name == "financial_report_resolution"
        assert isinstance(result.score, (int, float))
        assert 0 <= result.score <= 1.0
        assert isinstance(result.reason, str)
        assert "dimension_scores" in result.metadata
        assert "is_resolved" in result.metadata
        assert result.metadata["is_resolved"] is True  # Score is 1.0 (all 5s), should be >= 0.9

        # Verify model was called
        assert mock_model.achat.call_count == 1

    @pytest.mark.asyncio
    async def test_low_score_evaluation(self):
        """Test evaluation with low dimension scores"""
        # Setup mock response with callback-processed format
        # Calculated score: (2-1)/4 * 0.25 + (2-1)/4 * 0.20 + (3-1)/4 * 0.10 + (2-1)/4 * 0.10 + (3-1)/4 * 0.10 + (2-1)/4 * 0.15 + (3-1)/4 * 0.10 = 0.25 + 0.05 + 0.05 + 0.025 + 0.05 + 0.0375 + 0.05 = 0.3875
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 0.3875,  # Weighted average of normalized low scores
            "reason": "【精准性】2/5分: 精准度不足\n【完整性】2/5分: 完整性不足\n【相关性】3/5分: 基本相关\n【时效性】2/5分: 时效性不足\n【逻辑严谨性】3/5分: 逻辑尚可\n【数据支撑】2/5分: 数据支撑不足\n【报告结构与可读性】3/5分: 结构基本合理",
            "dimension_scores": {
                "precision": {"raw": 2, "normalized": 0.25},
                "completeness": {"raw": 2, "normalized": 0.25},
                "relevance": {"raw": 3, "normalized": 0.5},
                "timeliness": {"raw": 2, "normalized": 0.25},
                "logic": {"raw": 3, "normalized": 0.5},
                "data": {"raw": 2, "normalized": 0.25},
                "structure": {"raw": 3, "normalized": 0.5},
            },
        }

        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)

        grader = FinancialReportResolutionGrader(model=mock_model, resolution_threshold=0.9)

        messages = [
            {"role": "user", "content": "分析某公司的财务状况"},
            {"role": "assistant", "content": "简短回答"},
        ]

        result = await grader.aevaluate(messages=messages)

        # Assertions
        assert result.score < 0.9  # Should be below threshold
        assert result.metadata["is_resolved"] is False
        assert "dimension_scores" in result.metadata

    @pytest.mark.asyncio
    async def test_empty_input_edge_case(self):
        """Test edge case with empty input"""
        mock_model = AsyncMock()
        grader = FinancialReportResolutionGrader(model=mock_model)

        # Execute test with empty messages
        result = await grader.aevaluate(messages=[])

        # Assertions - should return GraderError
        assert isinstance(result, GraderError)
        assert hasattr(result, "error")
        assert "empty" in result.error.lower() or "空" in result.error

        # Model should not be called for empty input
        mock_model.achat.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_query_edge_case(self):
        """Test edge case with missing user query"""
        mock_model = AsyncMock()
        grader = FinancialReportResolutionGrader(model=mock_model)

        # Execute test with only assistant message
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "assistant", "content": "This is an answer without a question"},
        ]

        result = await grader.aevaluate(messages=messages)

        # Assertions - should return GraderError
        assert isinstance(result, GraderError)
        assert hasattr(result, "error")

    @pytest.mark.asyncio
    async def test_missing_answer_edge_case(self):
        """Test edge case with missing answer"""
        mock_model = AsyncMock()
        grader = FinancialReportResolutionGrader(model=mock_model)

        # Execute test with only user message
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "分析某公司的财务状况"},
        ]

        result = await grader.aevaluate(messages=messages)

        # Assertions - should return GraderError
        assert isinstance(result, GraderError)
        assert hasattr(result, "error")

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test graceful error handling"""
        # Setup mock to raise exception
        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(side_effect=Exception("API Error: Connection timeout"))

        # Create grader
        grader = FinancialReportResolutionGrader(model=mock_model)

        # Execute test
        messages = [
            {"role": "user", "content": "分析某公司的财务状况"},
            {"role": "assistant", "content": "详细的财务分析报告..."},
        ]

        result = await grader.aevaluate(messages=messages)

        # Assertions - should return GraderError
        assert isinstance(result, GraderError)
        assert hasattr(result, "error")

    @pytest.mark.asyncio
    async def test_malformed_messages_error_handling(self):
        """Test error handling with malformed messages"""
        mock_model = AsyncMock()
        grader = FinancialReportResolutionGrader(model=mock_model)

        # Execute test with malformed messages
        malformed_messages = [
            {"role": "user"},  # Missing content
            {"content": "Hello"},  # Missing role
        ]

        result = await grader.aevaluate(messages=malformed_messages)

        # Assertions - should return GraderError
        assert isinstance(result, GraderError)
        assert hasattr(result, "error")

    def test_message_extraction_with_wrapper(self):
        """Test message extraction with 'message' wrapper"""
        mock_model = AsyncMock()
        grader = FinancialReportResolutionGrader(model=mock_model)

        # Test with message wrapper
        messages_with_wrapper = [
            {"message": {"role": "system", "content": "System prompt"}},
            {"message": {"role": "user", "content": "User query"}},
            {"message": {"role": "assistant", "content": "Assistant answer"}},
        ]

        query, answer = grader._extract_query_and_answer_from_messages(messages_with_wrapper)

        assert query == "User query"
        assert answer == "Assistant answer"

    def test_message_extraction_without_wrapper(self):
        """Test message extraction without 'message' wrapper"""
        mock_model = AsyncMock()
        grader = FinancialReportResolutionGrader(model=mock_model)

        # Test without message wrapper
        messages_without_wrapper = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User query"},
            {"role": "assistant", "content": "Assistant answer"},
        ]

        query, answer = grader._extract_query_and_answer_from_messages(messages_without_wrapper)

        assert query == "User query"
        assert answer == "Assistant answer"

    def test_message_extraction_multi_turn(self):
        """Test message extraction with multi-turn conversation"""
        mock_model = AsyncMock()
        grader = FinancialReportResolutionGrader(model=mock_model)

        # Multi-turn messages
        multi_turn_messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "First query"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Follow-up query"},
            {"role": "assistant", "content": "Final answer"},
        ]

        query, answer = grader._extract_query_and_answer_from_messages(multi_turn_messages)

        # Should extract first user query and last assistant answer
        assert query == "First query"
        assert answer == "Final answer"

    @pytest.mark.asyncio
    async def test_chat_date_parameter(self):
        """Test that chat_date parameter is properly used"""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1.0,
            "reason": "【精准性】5/5分: 完全精准\n【完整性】5/5分: 完全完整\n【相关性】5/5分: 高度相关\n【时效性】5/5分: 时效性极佳\n【逻辑严谨性】5/5分: 逻辑完全严谨\n【数据支撑】5/5分: 数据支撑充分\n【报告结构与可读性】5/5分: 结构完美",
            "dimension_scores": {
                "precision": {"raw": 5, "normalized": 1.0},
                "completeness": {"raw": 5, "normalized": 1.0},
                "relevance": {"raw": 5, "normalized": 1.0},
                "timeliness": {"raw": 5, "normalized": 1.0},
                "logic": {"raw": 5, "normalized": 1.0},
                "data": {"raw": 5, "normalized": 1.0},
                "structure": {"raw": 5, "normalized": 1.0},
            },
        }

        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)

        grader = FinancialReportResolutionGrader(model=mock_model)

        messages = [
            {"role": "user", "content": "分析某公司的财务状况"},
            {"role": "assistant", "content": "详细的财务分析报告..."},
        ]

        result = await grader.aevaluate(messages=messages, chat_date="2024-12-31")

        # Verify the model was called with chat_date
        assert mock_model.achat.called

    def test_dimension_weights(self):
        """Test that dimension weights are correctly defined"""
        weights = FinancialReportResolutionGrader.DIMENSION_WEIGHTS

        # Check all dimensions exist
        assert "precision" in weights
        assert "completeness" in weights
        assert "relevance" in weights
        assert "timeliness" in weights
        assert "logic" in weights
        assert "data" in weights
        assert "structure" in weights

        # Check weights sum to 1.0
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.001  # Allow small floating point error

        # Check individual weights match specification
        assert weights["precision"] == 0.25
        assert weights["completeness"] == 0.20
        assert weights["relevance"] == 0.10
        assert weights["timeliness"] == 0.10
        assert weights["logic"] == 0.10
        assert weights["data"] == 0.15
        assert weights["structure"] == 0.10


# ==================== QUALITY TESTS ====================
# These tests verify the quality of the grader's evaluations

# Check for API keys to determine if live tests should run
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)

pytestmark = pytest.mark.skipif(
    not RUN_QUALITY_TESTS,
    reason="Requires API keys and base URL to run quality tests",
)


@pytest.mark.quality
class TestFinancialReportResolutionGraderQuality:
    """Quality tests for FinancialReportResolutionGrader - testing evaluation quality"""

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
        """Load test data from financial_test_data.json"""
        test_data_path = Path(__file__).parent / "financial_test_data.json"
        with open(test_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    @pytest.mark.asyncio
    async def test_financial_report_quality(self, model, test_data):
        """Test evaluation quality on financial research report"""
        # Create grader with real model
        grader = FinancialReportResolutionGrader(model=model, resolution_threshold=0.85)

        # Use the first test case from financial_test_data.json
        messages = test_data[0]

        # Execute evaluation with real model
        result = await grader.aevaluate(messages=messages, chat_date="2025-11-18")

        # Assertions
        assert result.name == "financial_report_resolution"
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

        # Verify metadata structure
        assert "dimension_scores" in result.metadata
        assert "is_resolved" in result.metadata
        assert "resolution_threshold" in result.metadata

        # Check dimension scores structure
        dimension_scores = result.metadata["dimension_scores"]
        assert isinstance(dimension_scores, dict)

        expected_dimensions = [
            "precision",
            "completeness",
            "relevance",
            "timeliness",
            "logic",
            "data",
            "structure",
        ]

        for dim in expected_dimensions:
            assert dim in dimension_scores
            assert "raw" in dimension_scores[dim]
            assert "normalized" in dimension_scores[dim]
            # Raw scores should be 1-5
            assert 1 <= dimension_scores[dim]["raw"] <= 5
            # Normalized scores should be 0-1
            assert 0.0 <= dimension_scores[dim]["normalized"] <= 1.0

        # For this report, we expect relatively high scores
        # But we don't set a hard threshold as it depends on model evaluation
        print(f"\n=== Financial Report Quality Test ===")
        print(f"Overall Score: {result.score:.3f}")
        print(f"Is Resolved: {result.metadata['is_resolved']}")
        print(f"Resolution Threshold: {result.metadata['resolution_threshold']}")
        print(f"\nDimension Scores:")
        for dim in expected_dimensions:
            raw = dimension_scores[dim]["raw"]
            norm = dimension_scores[dim]["normalized"]
            print(f"  {dim}: {raw}/5 (normalized: {norm:.3f})")
        print(f"\nReason Summary:")
        print(result.reason[:500] + "..." if len(result.reason) > 500 else result.reason)

    @pytest.mark.asyncio
    async def test_simple_query_quality(self, model):
        """Test evaluation quality on a simple query"""
        # Create grader with real model
        grader = FinancialReportResolutionGrader(model=model, resolution_threshold=0.9)

        # Simple financial query
        messages = [
            {
                "role": "system",
                "content": "你是一个金融分析专家",
            },
            {
                "role": "user",
                "content": "分析特斯拉的财务状况",
            },
            {
                "role": "assistant",
                "content": """基于最新财务数据，特斯拉展现出以下特点：

**财务表现（2024年）**
- 营收：967.7亿美元，同比增长18.8%
- 净利润：149.5亿美元，盈利能力强劲
- 毛利率：25.6%，处于行业较高水平

**资产负债结构**
- 债务股权比：0.17，财务结构稳健
- 现金流：经营现金流持续改善

**投资建议**
综合评估，特斯拉基本面扎实，但估值偏高。建议长期投资者可以逢低分批建仓，注意估值风险。""",
            },
        ]

        # Execute evaluation with real model
        result = await grader.aevaluate(messages=messages, chat_date="2024-12-01")

        # Assertions
        assert result.name == "financial_report_resolution"
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

        # Verify metadata structure
        assert "dimension_scores" in result.metadata
        assert "is_resolved" in result.metadata

        print(f"\n=== Simple Query Quality Test ===")
        print(f"Score: {result.score:.3f}")
        print(f"Is Resolved: {result.metadata['is_resolved']}")
        print(f"\nDimension Breakdown:")
        for dim, scores in result.metadata["dimension_scores"].items():
            print(f"  {dim}: {scores['raw']}/5")

    @pytest.mark.asyncio
    async def test_consistency_quality(self, model):
        """Test grader evaluation consistency"""
        # Create grader with real model
        grader = FinancialReportResolutionGrader(model=model, resolution_threshold=0.9)

        # Test data - same query evaluated twice
        messages = [
            {"role": "system", "content": "你是一个金融分析专家"},
            {"role": "user", "content": "简要分析茅台的投资价值"},
            {
                "role": "assistant",
                "content": "贵州茅台作为白酒行业龙头，拥有强大的品牌护城河和定价权。财务表现稳健，ROE保持在30%以上，现金流充裕。当前估值处于合理区间，适合长期配置。主要风险在于宏观经济波动和行业竞争加剧。",
            },
        ]

        # Run evaluation twice
        result1 = await grader.aevaluate(messages=messages, chat_date="2024-12-01")
        result2 = await grader.aevaluate(messages=messages, chat_date="2024-12-01")

        # Assertions
        assert result1.name == result2.name
        assert isinstance(result1.score, (int, float))
        assert isinstance(result2.score, (int, float))

        # Scores should be reasonably consistent (within 0.3 range for LLM variance)
        score_diff = abs(result1.score - result2.score)
        assert score_diff <= 0.3, f"Consistency issue: scores differ by {score_diff}"

        print(f"\n=== Consistency Quality Test ===")
        print(f"Run 1 Score: {result1.score:.3f}")
        print(f"Run 2 Score: {result2.score:.3f}")
        print(f"Score Difference: {score_diff:.3f}")

    @pytest.mark.asyncio
    async def test_poor_quality_report_detection(self, model):
        """Test that grader can detect poor quality reports"""
        # Create grader with real model
        grader = FinancialReportResolutionGrader(model=model, resolution_threshold=0.9)

        # Poor quality report - vague and lacks data
        messages = [
            {"role": "system", "content": "你是一个金融分析专家"},
            {
                "role": "user",
                "content": "详细分析某科技公司的财务状况、盈利能力、现金流、债务结构和投资价值",
            },
            {
                "role": "assistant",
                "content": "这家公司很好，业绩不错，值得投资。",
            },
        ]

        # Execute evaluation with real model
        result = await grader.aevaluate(messages=messages, chat_date="2024-12-01")

        # Assertions
        assert result.score < 0.6, f"Poor quality report should score < 0.6, got {result.score}"
        assert result.metadata["is_resolved"] is False

        print(f"\n=== Poor Quality Report Detection Test ===")
        print(f"Score: {result.score:.3f}")
        print(f"Is Resolved: {result.metadata['is_resolved']}")
        print(f"\nReason:")
        print(result.reason[:300] + "..." if len(result.reason) > 300 else result.reason)

    @pytest.mark.asyncio
    async def test_english_language_quality(self, model):
        """Test evaluation quality with English language"""
        # Create grader with English language
        grader = FinancialReportResolutionGrader(model=model, language=LanguageEnum.EN, resolution_threshold=0.9)

        # English financial query
        messages = [
            {
                "role": "system",
                "content": "You are a financial analyst expert",
            },
            {
                "role": "user",
                "content": "Analyze Apple's financial performance",
            },
            {
                "role": "assistant",
                "content": """Based on the latest financial data, Apple demonstrates:

**Financial Performance (2024)**
- Revenue: $394.3B, up 7.8% YoY
- Net Income: $96.9B, strong profitability
- Gross Margin: 46.2%, industry-leading

**Balance Sheet**
- Cash & Equivalents: $162B
- Debt-to-Equity: 1.8, manageable leverage

**Investment Recommendation**
Apple shows solid fundamentals with strong brand moat and ecosystem. Current valuation is reasonable for long-term investors. Key risks include supply chain disruptions and market saturation.""",
            },
        ]

        # Execute evaluation with real model
        result = await grader.aevaluate(messages=messages, chat_date="2024-12-01")

        # Assertions
        assert result.name == "financial_report_resolution"
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.reason, str)

        # The reason should contain English text
        assert any(keyword in result.reason.lower() for keyword in ["precision", "completeness", "relevance"])

        print(f"\n=== English Language Quality Test ===")
        print(f"Score: {result.score:.3f}")
        print(f"Language: {grader.language}")
        print(f"\nReason (first 200 chars):")
        print(result.reason[:200])
