"""
Rubrics-based Grader for Evaluating Agent Trajectory Performance - 基于评价标准(rubrics)的LLM评估器

This module provides a grader that evaluates agent trajectory performance based on
customizable evaluation rubrics (criteria with check points and weights).
"""

import textwrap
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field

from rm_gallery.core.graders.base_grader import GraderMode, GraderScore
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.graders.schema import GraderError
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.oai.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long


# Pydantic models for structured LLM output
class RubricEvaluation(BaseModel):
    """Single rubric evaluation result."""

    dimension: str = Field(description="评价维度名称/Name of the evaluation dimension")
    reasoning: str = Field(description="评价理由/Reasoning for the evaluation")
    check_points_passed: List[str] = Field(default=[], description="通过的检查点列表/List of check points that passed")
    check_points_failed: List[str] = Field(
        default=[], description="未通过的检查点列表/List of check points that failed"
    )


class RubricsEvaluationOutput(BaseModel):
    """Structured output model for rubrics evaluation."""

    dimension_evaluations: List[RubricEvaluation] = Field(
        description="各维度评估结果/Evaluation results for each dimension"
    )


# Chinese Prompt Template
RUBRICS_EVALUATION_PROMPT_ZH = """你是一个专业的Agent表现评估专家。你的任务是根据评价标准（Rubrics）对Agent的执行过程进行评估。

## 待评估内容

**用户问题：**
{query}

**Agent的工具调用：**
{tool_calls}

**Agent的最终回复：**
{final_response}

## 评价标准（Rubrics）

{rubrics}

## 评估任务

请针对上述**每个评价维度**，对Agent的表现进行独立评估：

1. **判断该维度的每个检查点是否通过**：
   - check_points_passed: 通过的检查点列表（将检查点完整内容复制到这里）
   - check_points_failed: 未通过的检查点列表（将检查点完整内容复制到这里）
   - 注意：每个检查点必须归类到passed或failed之一，不能遗漏
2. **给出该维度的评价理由**（简洁说明检查点的通过情况）

**重要提醒：**
- 你必须对上述所有评价维度都进行评估，不能遗漏任何维度
- 你的输出应该是评估结果（检查点判断、理由），而不是复述Agent的工具返回数据！
- 每个维度独立评估，检查点的通过情况是最重要的输出
- dimension字段必须与上述评价标准中的维度名称完全一致（包括大小写、标点符号）
- check_points_passed和check_points_failed中的检查点内容应与评价标准中的检查点完全一致

## 输出格式要求

**核心映射关系：上述评价标准中的每个维度 ➜ dimension_evaluations数组中的一个对象**

### 输出结构示例：

```
{{
  "dimension_evaluations": [
    {{
      "dimension": "评价维度1的名称（必须与上述评价标准中的维度名称完全一致）",
      "reasoning": "针对这个维度的评估理由，说明检查点通过情况",
      "check_points_passed": ["通过的检查点1的完整内容", "通过的检查点2的完整内容"],
      "check_points_failed": ["未通过的检查点1的完整内容"]
    }},
    {{
      "dimension": "评价维度2的名称",
      "reasoning": "针对这个维度的评估理由",
      "check_points_passed": ["通过的检查点的完整内容"],
      "check_points_failed": []
    }}
    // ... 上述评价标准有多少个维度，这里必须有对应数量的对象，一一对应
  ]
}}
```

**输出要求：**
1. dimension_evaluations必须是数组，包含所有评价维度，每个维度对应一个对象
2. 每个维度对象必须包含：dimension、reasoning、check_points_passed、check_points_failed
3. dimension字段必须与评价标准中的维度名称完全一致
4. check_points_passed和check_points_failed必须包含完整的检查点内容（与评价标准一致）
5. 不要在JSON顶层添加维度名称作为键
6. 不要复述原始数据，只输出评估结果
7. 无需输出score字段，分数将根据检查点通过率自动计算
8. 严格遵守上述JSON格式示例

JSON：
"""

# English Prompt Template
RUBRICS_EVALUATION_PROMPT_EN = """You are a professional Agent performance assessment expert. Your task is to evaluate the Agent's execution process based on evaluation criteria (Rubrics).

## Content to Evaluate

**User Question:**
{query}

**Agent's Tool Calls:**
{tool_calls}

**Agent's Final Response:**
{final_response}

## Evaluation Criteria (Rubrics)

{rubrics}

## Evaluation Task

Please independently evaluate the Agent's performance for **each evaluation dimension**:

1. **Determine whether each check point in the dimension passes**:
   - check_points_passed: List of passed check points (copy the complete content of check points here)
   - check_points_failed: List of failed check points (copy the complete content of check points here)
   - Note: Each check point must be classified into either passed or failed, none can be omitted
2. **Provide evaluation reasoning for the dimension** (briefly explain the check point passing situation)

**Important Reminders:**
- You must evaluate all evaluation dimensions mentioned above, none can be omitted
- Your output should be evaluation results (check point judgments, reasoning), not a repetition of the Agent's tool return data!
- Each dimension is evaluated independently, the passing status of check points is the most important output
- The dimension field must exactly match the dimension name in the evaluation criteria above (including case and punctuation)
- The content in check_points_passed and check_points_failed should exactly match the check points in the evaluation criteria

## Output Format Requirements

**Core Mapping: Each dimension in the evaluation criteria above ➜ One object in the dimension_evaluations array**

### Output Structure Example:

```
{{
  "dimension_evaluations": [
    {{
      "dimension": "Name of evaluation dimension 1 (must exactly match the dimension name in evaluation criteria)",
      "reasoning": "Evaluation reasoning for this dimension, explaining check point passing status",
      "check_points_passed": ["Complete content of passed check point 1", "Complete content of passed check point 2"],
      "check_points_failed": ["Complete content of failed check point 1"]
    }},
    {{
      "dimension": "Name of evaluation dimension 2",
      "reasoning": "Evaluation reasoning for this dimension",
      "check_points_passed": ["Complete content of passed check point"],
      "check_points_failed": []
    }}
    // ... The evaluation criteria has multiple dimensions, there must be corresponding objects here, one-to-one correspondence
  ]
}}
```

**Output Requirements:**
1. dimension_evaluations must be an array containing all evaluation dimensions, one object per dimension
2. Each dimension object must include: dimension, reasoning, check_points_passed, check_points_failed
3. The dimension field must exactly match the dimension name in the evaluation criteria
4. check_points_passed and check_points_failed must contain complete check point content (matching the evaluation criteria)
5. Do not add dimension names as keys at the top level of JSON
6. Do not repeat raw data, only output evaluation results
7. No need to output score field, scores will be automatically calculated based on check point pass rate
8. Strictly follow the JSON format example above

JSON:
"""

# Build default template from prompts
DEFAULT_RUBRICS_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(RUBRICS_EVALUATION_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(RUBRICS_EVALUATION_PROMPT_ZH),
            ),
        ],
    },
)


class RubricsBasedTrajectoryPerformance(LLMGrader):
    """
    Rubrics-based grader for evaluating agent performance.

    This grader evaluates agent responses based on customizable evaluation rubrics.
    Each rubric contains multiple dimensions (criteria), and each dimension has:
    - dimension: Name of the evaluation dimension
    - description: Description of what to evaluate
    - check_points: List of specific check points to verify

    The grader uses LLM to evaluate each check point as passed or failed,
    then calculates dimension scores based on pass rate, and computes a
    weighted total score across all dimensions.

    Score calculation:
    1. For each dimension: score = passed_checks / total_checks
    2. Total score = weighted average of dimension scores (0.0-1.0)

    Attributes:
        name: Grader name
        model: ChatModelBase instance for evaluation
        language: Language for evaluation prompts

    Example:
        >>> from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
        >>> model = OpenAIChatModel(api_key="...", model="gpt-4o")
        >>> weights = {"完整性": 0.5, "准确性": 0.5}
        >>> grader = RubricsBasedTrajectoryPerformance(model=model)
        >>> rubrics = [
        ...     {
        ...         "dimension": "完整性",
        ...         "description": "报告是否完整覆盖所有关键点",
        ...         "check_points": ["覆盖所有关键指标", "包含时间信息"]
        ...     },
        ...     {
        ...         "dimension": "准确性",
        ...         "description": "数据是否准确",
        ...         "check_points": ["数值正确", "来源可靠"]
        ...     }
        ... ]
        >>> result = await grader.aevaluate(
        ...     messages=[
        ...         {"role": "user", "content": "分析公司财报"},
        ...         {"role": "assistant", "tool_calls": [...]},
        ...         {"role": "tool", "content": "财报数据..."},
        ...         {"role": "assistant", "content": "分析报告..."}
        ...     ],
        ...     rubrics=rubrics
        ... )
        >>> print(f"Score: {result.score}")  # 0.0-1.0
    """

    @staticmethod
    def _create_rubrics_callback(
        language: LanguageEnum = LanguageEnum.ZH,
    ) -> Callable[[Any], Dict[str, Any]]:
        """
        Create a callback function to process LLM evaluation output into final score.

        This callback:
        1. Extracts dimension evaluations from LLM response
        2. Calculates dimension scores based on check point pass rates
        3. Computes weighted total score across dimensions
        4. Generates detailed reason string

        Args:
            language: Language for generating the reason string

        Returns:
            Callable that processes ChatResponse into parsed dict with score and reason
        """

        def callback(response: Any) -> Dict[str, Any]:
            # Extract dimension evaluations from response
            dimension_evaluations_raw = []
            if response.parsed is not None:
                # Handle both nested and flat structures
                if "dimension_evaluations" in response.parsed:
                    dimension_evaluations_raw = response.parsed["dimension_evaluations"]
                else:
                    dimension_evaluations_raw = response.parsed

            # Parse into RubricEvaluation objects
            dimension_evaluations: List[RubricEvaluation] = []
            if isinstance(dimension_evaluations_raw, list):
                dimension_evaluations = [RubricEvaluation(**d) for d in dimension_evaluations_raw]
            else:
                logger.warning("Invalid dimension_evaluations format, expected list")

            # If no evaluations, return zero score
            if not dimension_evaluations:
                return {
                    "score": 0.0,
                    "reason": (
                        "未返回任何维度评估结果，得分为0。"
                        if language == LanguageEnum.ZH
                        else "No dimension evaluations returned, score is 0."
                    ),
                    "dimension_scores": {},
                    "dimension_evaluations": [],
                }

            # Calculate dimension scores based on check point pass rates
            dimension_scores = {}
            for eval_item in dimension_evaluations:
                passed_count = len(eval_item.check_points_passed)
                failed_count = len(eval_item.check_points_failed)
                total_count = passed_count + failed_count

                if total_count > 0:
                    score = passed_count / total_count
                else:
                    logger.warning(
                        f"维度 '{eval_item.dimension}' 没有任何检查点被评估（passed=0, failed=0），默认分数=0"
                    )
                    score = 0.0

                dimension_scores[eval_item.dimension] = score
                logger.debug(f"维度 '{eval_item.dimension}' 自动计算分数: {passed_count}/{total_count} = {score:.4f}")

            # Calculate average score across all dimensions (equal weight)
            if dimension_scores:
                normalized_score = sum(dimension_scores.values()) / len(dimension_scores)
            else:
                logger.warning("没有任何维度分数，返回0分")
                normalized_score = 0.0

            # Ensure score is in [0, 1] range
            normalized_score = max(0.0, min(1.0, normalized_score))

            logger.debug(
                f"平均总分计算: {sum(dimension_scores.values()):.4f} / {len(dimension_scores)} = {normalized_score:.4f}"
            )

            # Generate reason string
            if language == LanguageEnum.ZH:
                reason_lines = [f"总分: {normalized_score:.4f}\n"]
                for eval_item in dimension_evaluations:
                    dim_score = dimension_scores.get(eval_item.dimension, 0.0)
                    passed = len(eval_item.check_points_passed)
                    failed = len(eval_item.check_points_failed)
                    reason_lines.append(
                        f"【{eval_item.dimension}】分数: {dim_score:.2f} "
                        f"(通过: {passed}/{passed+failed})\n  理由: {eval_item.reasoning}"
                    )
                reason = "\n".join(reason_lines)
            else:
                reason_lines = [f"Total Score: {normalized_score:.4f}\n"]
                for eval_item in dimension_evaluations:
                    dim_score = dimension_scores.get(eval_item.dimension, 0.0)
                    passed = len(eval_item.check_points_passed)
                    failed = len(eval_item.check_points_failed)
                    reason_lines.append(
                        f"【{eval_item.dimension}】Score: {dim_score:.2f} "
                        f"(Passed: {passed}/{passed+failed})\n  Reasoning: {eval_item.reasoning}"
                    )
                reason = "\n".join(reason_lines)

            # Build detailed evaluation info
            dimension_evals_dict = [
                {
                    "dimension": eval_item.dimension,
                    "score": dimension_scores.get(eval_item.dimension, 0.0),
                    "reasoning": eval_item.reasoning,
                    "check_points_passed": eval_item.check_points_passed,
                    "check_points_failed": eval_item.check_points_failed,
                }
                for eval_item in dimension_evaluations
            ]

            return {
                "score": normalized_score,
                "reason": reason,
                "dimension_scores": dimension_scores,
                "dimension_evaluations": dimension_evals_dict,
            }

        return callback

    def __init__(
        self,
        model: Union[BaseChatModel, dict],
        template: Optional[PromptTemplate] = DEFAULT_RUBRICS_TEMPLATE,
        language: LanguageEnum = LanguageEnum.ZH,
    ):
        """
        Initialize the RubricsBasedTrajectoryPerformance.

        Args:
            model (Union[BaseChatModel, dict]): The chat model to use for evaluation.
                Can be either a BaseChatModel instance or a dictionary configuration.
            template (Optional[PromptTemplate]): The prompt template for rubrics evaluation.
                Defaults to DEFAULT_RUBRICS_TEMPLATE.
            language (LanguageEnum): Language for the evaluation prompt.
                Defaults to LanguageEnum.ZH (Chinese).

        Example:
            >>> from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
            >>> model = OpenAIChatModel(api_key="...", model="gpt-4o")
        """

        super().__init__(
            name="rubrics_based_trajectory_performance",
            mode=GraderMode.POINTWISE,
            description="Rubrics-based evaluation for agent performance",
            model=model,
            template=template,
            language=language,
            structured_model=RubricsEvaluationOutput,
            callback=self._create_rubrics_callback(language=language),
        )

    def _extract_info_from_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> tuple[str, str, str]:
        """
        Extract user query, tool calls info, and final response from messages.

        Args:
            messages: List of message dicts (standard format).

        Returns:
            Tuple of (query, tool_calls_str, final_response)
        """
        # Filter out system messages and unwrap nested structure
        messages = [msg.get("message", msg) for msg in messages]
        non_system_messages = [msg for msg in messages if msg.get("role", "") != "system"]

        if not non_system_messages:
            return "", "", ""

        # Extract user query (first non-system user message)
        query = ""
        if non_system_messages[0].get("role", "") == "user":
            query = non_system_messages[0].get("content", "")

        # Extract final response (last assistant message content)
        final_response = ""
        for msg in reversed(non_system_messages):
            if msg.get("role", "") == "assistant" and msg.get("content", ""):
                final_response = msg.get("content", "")
                break

        # Extract tool calls information (simplified)
        tool_calls_info = []
        tool_call_count = 0
        for msg in non_system_messages:
            if msg.get("role", "") == "assistant" and msg.get("tool_calls"):
                # Extract tool call information
                for tool_call in msg.get("tool_calls", []):
                    tool_call_count += 1
                    if isinstance(tool_call, dict):
                        function_info = tool_call.get("function", {})
                        tool_name = function_info.get("name", "unknown")
                        # Get parameter keys only (not values, to keep it concise)
                        arguments = function_info.get("arguments", {})
                        if isinstance(arguments, str):
                            import json

                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                arguments = {}
                        param_keys = list(arguments.keys()) if isinstance(arguments, dict) else []
                        tool_calls_info.append(f"{tool_call_count}. {tool_name}(参数: {', '.join(param_keys)})")

            # Also extract tool responses (simplified)
            if msg.get("role", "") == "tool" and msg.get("content"):
                content_brief = msg.get("content", "")[:150].replace("\n", " ").strip() + "..."
                if tool_calls_info:
                    tool_calls_info[-1] += f"\n   结果摘要: {content_brief}"

        tool_calls_str = "\n".join(tool_calls_info) if tool_calls_info else "无工具调用"

        return query, tool_calls_str, final_response

    async def aevaluate(
        self,
        messages: List[Dict[str, Any]],
        rubrics: List[Dict[str, Any]],
    ) -> GraderScore:
        """
        Evaluate agent performance based on rubrics.

        The evaluation:
        1. Extracts information from messages (query, tool calls, final response)
        2. Uses LLM to evaluate each check point in rubrics as passed/failed
        3. Calculates dimension scores based on check point pass rates
        4. Computes weighted total score across dimensions (0.0-1.0)

        Args:
            messages: List of messages (standard format, including system, user, assistant, tool)
                The "message" key for message can be optional.

                Example:
                ```
                [
                  {"role": "user", "content": "分析我的持仓"},
                  {"role": "assistant", "tool_calls": [{"function": {"name": "search", ...}}]},
                  {"role": "tool", "name": "search", "content": "用户持仓数据..."},
                  {"role": "assistant", "content": "您的持仓分析报告..."}
                ]
                ```

            rubrics: List of evaluation criteria (dimensions).
                Structure:
                ```
                [
                    {
                        "dimension": "维度名称",
                        "description": "评估标准描述",
                        "check_points": ["检查点1", "检查点2"]
                    },
                    ...
                ]
                ```

        Returns:
            GraderScore: Rubrics evaluation score
                - score: Weighted total score (0.0-1.0)
                - reason: Formatted string with dimension scores and reasoning
                - metadata: Contains dimension_scores, dimension_evaluations

        Example:
            >>> result = await grader.aevaluate(
            ...     messages=[
            ...         {"role": "user", "content": "分析某公司财报"},
            ...         {"role": "assistant", "content": "分析报告..."}
            ...     ],
            ...     rubrics=[
            ...         {
            ...             "dimension": "完整性",
            ...             "description": "报告完整性",
            ...             "check_points": ["包含所有关键指标"],
            ...             "weight": 1.0
            ...         }
            ...     ]
            ... )
            >>> print(f"Rubrics Score: {result.score}")
        """
        # Validate rubrics
        if not rubrics or not isinstance(rubrics, list):
            logger.warning("Invalid or empty rubrics, must be a list")
            return GraderError(name=self.name, error="Invalid or empty rubrics, must be a list")

        # Extract information from messages
        query, tool_calls_str, final_response = self._extract_info_from_messages(messages)

        if not query or not tool_calls_str or not final_response:
            logger.warning("Empty query or tool_calls or final response, returning error")
            return GraderError(name=self.name, error="Empty query or tool_calls or final_response")

        try:
            # Convert rubrics to string for prompt
            rubrics_str = str(rubrics)

            # Call parent evaluation with rubrics string
            result = await super().aevaluate(
                query=query,
                tool_calls=tool_calls_str,
                final_response=final_response,
                rubrics=rubrics_str,
            )

            # Check if result is a GraderError, and if so, return it directly
            if isinstance(result, GraderError):
                return result

            return GraderScore(
                name=self.name,
                score=result.score,
                reason=result.reason,
                metadata=result.metadata,
            )

        except Exception as e:
            logger.error(f"Error evaluating {self.name}: {e}")
            return GraderError(name=self.name, error=str(e))
