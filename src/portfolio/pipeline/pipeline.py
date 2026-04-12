from __future__ import annotations

import inspect
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

from src.exceptions import InvalidStateError


class BaseStep(ABC):
    """
    すべてのStepが継承する中小規定クラス
    - execute() は必ず override する
    """

    def __init__(self) -> None:
        self.log_out: Optional[str] = None

    @abstractmethod
    def execute(self, state: StateLike, **kwargs:Any) -> StateLike:
        """stepのエントリポイント"""
        pass

    def __call__(self, state: StateLike, **kwargs: Any) -> StateLike:
        return self.execute(state, **kwargs)

    def __rshift__(self, other: Union[BaseStep, Pipeline]) -> Pipeline:
        if isinstance(other, Pipeline):
            return Pipeline([self, *other.steps])
        elif isinstance(other, BaseStep):
            return Pipeline([self, other])
        else:
            raise TypeError(f"Unsupported type:{type(other)}")

    def _require_state(
        self,
        state: Any,
        *,
        allowed_types: Optional[Tuple[type, ...]] = (
            TrainState,
            PredictState,
        ),
        required_attr: Iterable[str] = (),
        required_non_none: Iterable[str] = (),
    ) -> None:
        """
        stepの入り口でstate形状を検証する。
        - stateがNoneの場合は即エラー
        - allowed_typesが指定されていて型が一致しない場合はエラー
        - required_attrsの属性がない場合はエラー
        - required_non_noneの属性がNoneの場合はエラー
        """
        step_name = self.__class__.__name__

        if state is None:
            raise InvalidStateError(f"{step_name}: stateがNoneです")

        if allowed_types is not None and not isinstance(state, allowed_types):
            raise TypeError(
                f"{step_name}: stateの型が想定外です"
                f"(expected={allowed_types}, actual={type(state)})"
            )

        for attr in required_attrs:
            if not hasattr(state, attr):
                raise TypeError(f"{step_name}: stateに{attr}属性がありません")

        for attr in required_non_none:
            if getattr(state, attr, None) is None:
                raise InvalidStateError(f"{step_name}: state.{attr}がNoneです")

    def _gen_next_prefix(self, prefix: str) -> str:
        pattern = re.compile(r"\[(\d+(?:\.\d+)*)\]")

        def repl(match: re.Match) -> str:
            content = match.group(1)

            if "." in content:
                parts = content.split(".")
                parts[-1] = str(int(parts[-1]) + 1)
                new_content = ".".join(parts)
            else:
                new_content = str(int(content) + 1)
            
            return f"[{new_content}]"

        return pattern.sub(repl, prefix, count=1)

    def depict(self, prefix: str = "0") -> Tuple[List[str], str]:
        """
        stepを描画するメソッド

        args:
            prefix: ノードカウントのプレフィックス（例："1", "1.2"）

        return:
            （行のリスト, 次のプレフィックス）
        """
        label = self.__class__.__name__
        lines = [f"{prefix} {label}"]
        return lines, self.__gen__next__prefix(prefix)


class BranchStep(BaseStep):
    def __init__(
        self, 
        condition_fn: Callable[[StateLike], bool],
        true_steps: Union[BaseSteps, Pipeline],
        false_steps: Union[BaseStep, Pipeline],
    ):
        super().__init__()
        self.condition_fn = condition_fn
        self.true_steps = true_steps
        self.false_steps = faslse_steps

    def _extract_lambda_content(self, fn: Callable) -> str:
        """
        ラムダ関数の中身を解析して、呼び出されている関数名と引数を抽出する
        例：lamda state: self.check_existance(state, "Feature") -> check_existance("Feature")
        """
        if hasattr(fn, "__name__") and fn.__name__ != "<lambda>":
            return fn.__name__
        
        try:
            source = inspect.getsource(fn)
            # self.method_name(state, "arg") または self.method_name(state, 'arg')のパターン抽出
            match = re.search(r'self\.(\w+)\([^,)]+,\s*["\']([^"\']+)["\']\)', source)
            if match:
                return f'{match.group(1)}("{match.group(2)}")'
        except (OSError, TypeError):
            pass

        return str(fn)

    def execute(self, state: StateLike, **kwargs: Any) -> StateLike:
        self._require_state(state)
        # どちらのルートに進むか決定
        steps = self.true_steps if self.condition_fn(state) else self.false_steps

        # ルート内のstepを順番に実行
        if isinstance(steps, Pipeline):
            return steps.run(state)
        else:
            return steps(state)
        
        return state

    def depict(self, prefix: str = "0") -> Tuple[list[str], str]:
        condition_name = self._extract_lambda_cotent(self.condition_fn)
        lines = [f"{prefix} BranchStep: \033[33m{condition_name}\033[0m"]
        spaces = "".join(c for c in prefix if c == " ")
        lines.append(f"{space}    \033[32mTrue:\033[0m")

        true_prefix = "      " + re.sub(r"(\d)(?!.*\d)", r"\1.1", prefix)
        true_lines, true_next_prefix = self.true_steps.depict(true_prefix)
        lines.etend(true_lines)

        lines.append(f"{spaces}    \033[31mFalse:\033[0m")
        false_prefix = "      " + re.sub(r"(\d)(?!.*\d)", r"\1.1", prefix)
        false_lines, false_next_prefix = self.flase_steps.depict(false_prefix)
        lines.extend(false_lines)

        return lines, self._gen_next_prefix(prefix)

    logger = logging.getLogger(__name__)


    class PipeLine:
        def __init__(slf, steps: List[BranchSteps], verbose: bool = False):
            self.steps = steps
            self.verbose = verboseself.step_counter = 0

        def __iter__(self) -> Iterator[BaseStep]:
            return iter(self.steps)

        def _next_step_index(self) -> int:
            idx = self.step_counter
            self.step_counter += 1
            return idx

        def __rshift__(self, other: Union[BaseStep, Pipeline]) -> Pipeline:
            if isinstance(other, BaseStep):
                return Pipeline(self.steps + [other], verbose=self.verbose)
            elif isinstance(other, Pipeline):
                return Pipeline(self.steps + other.steps, verbose=self.verbose)
            else:
                raise TypeError(f"Unsupported type: {type(other)}")
        
    def _execute_step_with_logging(
        self,
        step: BaseStep,
        state: StateLike,
        step_index: int,
        parent_step: Optional[str] = None,
    ) -> StateLike:
        """ステップを実行し、実行状況をログ出力"""
        step_name = step.__class__.__name__
        step_start = time.time()

        try:
            # BranchStepの場合は、内部のステップも個別にログ出力
            if isinstance(step, BranchStep):
                # 条件を評価してどちらのパスを実行するか決定
                condition_result = step.condition_fn(state)
                steps_to_execute = (
                    step.true_steps if condition_result else step.false_steps
                )

                # BranchStep全体の開始時間
                branch_start = time.time()

                if self.verbose:
                    logger.info(
                        f"[{step_index + 1}] {step_name} ({'True' if condition_result else 'False'})"
                    )

                # 内部のステップを実行
                if isinstance(steps_to_execute, Pipeline):
                    # Pipelineの場合は、各ステップを個別に実行
                    for _sub_idx, sub_step in enumerate(steps_to_execute.steps):
                        sub_step_index = self._next_step_index()
                        state = self._execute_step_with_logging(
                            sub_step, state, sub_step_index, parent_step=step_name
                        )
                elif isinstance(steps_to_execute, BaseStep):
                    # 単一のステップの場合
                    sub_step_index = self._next_step_index()
                    state = self._execute_step_with_logging(
                        steps_to_execute,
                        state,
                        sub_step_index,
                        parent_step=step_name,
                    )
                # 空のPipelineの場合は何もしない

                branch_end = time.time()
                branch_time = branch_end - branch_start

                if self.verbose:
                    logger.info(f"  [{step_index + 1}] {step_name}: {branch_time:.3f}s")

            else:
                # 通常のステップの場合
                result_state = step(state)
                state = result_state  # type: ignore
                step_end = time.time()
                execution_time = step_end - step_start

                if self.verbose:
                    indent = "  " if parent_step else ""
                    log_suffix = f" - {step.log_out}" if step.log_out else ""
                    logger.info(
                        f"{indent}[{step_index + 1}] {step_name}: {execution_time:.3f}s{log_suffix}"
                    )

            return state

        except Exception as e:
            step_end = time.time()
            execution_time = step_end - step_start

            error_msg = str(e)
            if self.verbose:
                indent = "  " if parent_step else ""
                log_suffix = f" - {step.log_out}" if step.log_out else ""
                logger.error(
                    f"{indent}[{step_index + 1}] {step_name}: {execution_time:.3f}s{log_suffix} (エラー: {error_msg})"
                )
            raise

    def run(self, state: StateLike) -> StateLike:
        """パイプラインを実行"""
        if self.verbose:
            logger.info("パイプラインを実行中...")
            logger.info("-" * 80)

        self.step_counter = 0
        start_time = time.time()

        for step in self.steps:
            try:
                step_index = self._next_step_index()
                state = self._execute_step_with_logging(step, state, step_index)
            except Exception as e:
                if self.verbose:
                    logger.error(f"パイプライン実行中にエラーが発生しました: {e}")
                raise

        if self.verbose:
            total_time = time.time() - start_time
            logger.info("-" * 80)
            logger.info(f"総実行時間: {total_time:.3f}s")

        return state

    def depict(self, prefix: str = "[0]") -> tuple[list[str], str]:
        lines = []
        current_prefix = prefix
        for step in self.steps:
            step_lines, next_prefix = step.depict(current_prefix)
            lines.extend(step_lines)
            current_prefix = next_prefix
        return lines, current_prefix

    def depict_str(self) -> str:
        """文字列としてパイプラインを描画する（後方互換性のため）"""
        lines, _ = self.depict("[1]")
        return "\n".join(["\033[1mPipeline: \033[0m\n"] + lines)