"""
Deep Research Agent Evaluation Tutorial

This tutorial demonstrates how to evaluate a deep research agent using multiple graders
that assess different aspects of the agent's performance:

1. Financial Report Resolution: Evaluates report quality and problem resolution
2. Financial Trajectory Faithfulness: Checks factual accuracy against search results
3. Rubrics-Based Performance: Evaluates against custom criteria
4. Trajectory Comprehensive: Assesses step-by-step contribution
5. Observation Information Gain: Measures information redundancy
6. Action Loop Detection: Detects repetitive actions

All graders are aggregated using GradingRunner for concurrent evaluation.
"""

import asyncio
from typing import Any, Dict, List

from tutorials.deep_research.graders.financial_report_resolution import FinancialReportResolutionGrader
from tutorials.deep_research.graders.financial_trajectory_faithfulness import FinancialTrajectoryFaithfulGrader
from rm_gallery.core.graders.agent.trajectory.rubrics_based_trajectory_performance import RubricsBasedTrajectoryPerformance
from rm_gallery.core.graders.agent.observation.observation_information_gain import ObservationInformationGainGrader
from rm_gallery.core.graders.agent.action.action_loop import ActionLoopDetectionGrader
from rm_gallery.core.graders.agent.trajectory.trajectory_comprehensive import TrajectoryComprehensiveGrader
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.models.schema.prompt_template import LanguageEnum
from rm_gallery.core.runner.grading_runner import GraderConfig, GradingRunner


def create_sample_data() -> List[Dict[str, Any]]:
    """Create sample evaluation data for demonstration.
    
    Returns:
        List of evaluation samples with agent trajectories
    """
    return [
        {
            "messages": [
                {"role": "user", "content": "分析贵州茅台2025年上半年的财务表现"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_financial_data",
                                "arguments": '{"company": "贵州茅台", "period": "2025H1"}',
                            }
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "截至2025年6月30日，贵州茅台实现营业收入789.5亿元，同比增长15.2%；净利润454.03亿元，同比增长18.5%。",
                },
                {
                    "role": "assistant",
                    "content": "根据财务数据，贵州茅台2025年上半年表现优异。营业收入达789.5亿元，同比增长15.2%；净利润454.03亿元，同比增长18.5%。公司盈利能力持续增强，业绩增长稳健。",
                },
            ],
            "chat_date": "2025-07-15",
            "rubrics": [
                {
                    "dimension": "数据准确性",
                    "description": "报告中的数据是否准确无误",
                    "check_points": ["营业收入数据正确", "净利润数据正确", "增长率计算准确"],
                },
                {
                    "dimension": "分析完整性",
                    "description": "分析是否全面覆盖关键指标",
                    "check_points": ["包含收入分析", "包含利润分析", "包含同比增长"],
                },
            ],
        }
    ]


def create_grader_configs(model: OpenAIChatModel, language: LanguageEnum = LanguageEnum.ZH) -> Dict[str, GraderConfig]:
    """Create grader configurations with appropriate mappers.
    
    Args:
        model: LLM model for evaluation
        language: Language for evaluation prompts
        
    Returns:
        Dictionary of grader configurations
    """
    return {
        # Report quality evaluation - requires messages and chat_date
        "report_resolution": GraderConfig(
            grader=FinancialReportResolutionGrader(model=model, language=language),
            mapper=lambda data: {"messages": data["messages"], "chat_date": data.get("chat_date")},
        ),
        # Factual accuracy evaluation - requires messages only
        "trajectory_faithfulness": GraderConfig(
            grader=FinancialTrajectoryFaithfulGrader(model=model, language=language),
            mapper=lambda data: {"messages": data["messages"]},
        ),
        # Rubrics-based evaluation - requires messages and rubrics
        "rubrics_performance": GraderConfig(
            grader=RubricsBasedTrajectoryPerformance(model=model, language=language),
            mapper=lambda data: {"messages": data["messages"], "rubrics": data.get("rubrics", [])},
        ),
        # Comprehensive trajectory evaluation - requires messages only
        "trajectory_comprehensive": GraderConfig(
            grader=TrajectoryComprehensiveGrader(model=model, language=language),
            mapper=lambda data: {"messages": data["messages"]},
        ),
        # Information gain evaluation - requires messages only
        "information_gain": GraderConfig(
            grader=ObservationInformationGainGrader(similarity_threshold=0.5),
            mapper=lambda data: {"messages": data["messages"]},
        ),
        # Action loop detection - requires messages only
        "action_loop": GraderConfig(
            grader=ActionLoopDetectionGrader(similarity_threshold=1.0),
            mapper=lambda data: {"messages": data["messages"]},
        ),
    }


async def main():
    """Main evaluation workflow."""
    # Initialize LLM model
    model = OpenAIChatModel(
        model="qwen3-max",
        temperature=0.0,
    )

    # Create grader configurations
    grader_configs = create_grader_configs(model, language=LanguageEnum.ZH)

    # Initialize runner with concurrent execution
    runner = GradingRunner(
        grader_configs=grader_configs,
        max_concurrency=6,  # Run all 6 graders concurrently
        show_progress=True,
    )

    # Prepare evaluation data
    dataset = create_sample_data()

    # Run evaluation
    print("Starting deep research agent evaluation...")
    results = await runner.arun(dataset)

    # Display results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    for grader_name, grader_results in results.items():
        print(f"\n{'─' * 80}")
        print(f"📊 {grader_name.upper()}")
        print(f"{'─' * 80}")

        for i, result in enumerate(grader_results):
            print(f"\nSample {i + 1}:")
            if hasattr(result, "score"):
                print(f"  Score: {result.score:.4f}")
                print(f"  Reason: {result.reason[:200]}...")
                if hasattr(result, "metadata") and result.metadata:
                    print(f"  Metadata: {list(result.metadata.keys())}")
            else:
                print(f"  Error: {result.error}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

