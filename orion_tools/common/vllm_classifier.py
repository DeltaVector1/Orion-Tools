"""vLLM-based text classifier."""
import re
from typing import List, Dict, Optional

from vllm import LLM, SamplingParams


class VLLMClassifier:
    """Wrapper for vLLM-based text classification."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
    ):
        self.model_name = model_name
        self.engine = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
        )

    @staticmethod
    def _build_prompt(text: str, labels: List[str], system_hint: Optional[str] = None, instruction: Optional[str] = None) -> str:
        label_str = ", ".join(labels)
        parts = []
        if system_hint:
            parts.append(system_hint.strip())
        if instruction:
            parts.append(instruction.strip())
        parts.append(
            f"Classify the following text into exactly one of the labels [{label_str}].\nText: {text}\nLabel:"
        )
        return "\n\n".join(parts)

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def classify_texts(
        self,
        texts: List[str],
        labels: List[str],
        system_hint: Optional[str] = None,
        instruction: Optional[str] = None,
        max_tokens: int = 3,
        temperature: float = 0.0,
        batch_size: int = 64,
    ) -> List[Dict[str, float]]:
        """
        Classify a list of texts.
        
        Returns:
            List of dicts with 'label' and 'score' keys
        """
        if not texts:
            return []
        normalized_allowed = {self._normalize(l): l for l in labels}
        sampling = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["\n"],
            logprobs=max(1, len(labels)),
        )
        prompts = [self._build_prompt(t, labels, system_hint, instruction) for t in texts]
        results: List[Dict[str, float]] = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            outputs = self.engine.generate(batch_prompts, sampling)
            for out in outputs:
                generated_text = out.outputs[0].text if out.outputs else ""
                norm = self._normalize(generated_text)
                matched_label = None
                for key, canonical in normalized_allowed.items():
                    if norm.startswith(key):
                        matched_label = canonical
                        break
                # Estimate confidence from first token logprob if available
                confidence = 1.0
                try:
                    first_token_logprob = None
                    if out.outputs and out.outputs[0].logprobs and out.outputs[0].logprobs[0]:
                        first_token_logprob = max(tok.logprob for tok in out.outputs[0].logprobs[0])
                        confidence = float(pow(2.718281828, first_token_logprob))
                except Exception:
                    pass
                if matched_label is None:
                    # Fallback: choose the first label
                    matched_label = labels[0]
                results.append({"label": matched_label, "score": confidence})
        return results
