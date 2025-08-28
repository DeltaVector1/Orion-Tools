import re
from typing import List, Dict, Optional

from vllm import LLM, SamplingParams


class VLLMClassifier:

	def __init__(
		self,
		model_name: str,
		tensor_parallel_size: int = 1,
		max_model_len: Optional[int] = None,
	):
		self.engine = LLM(
			model=model_name,
			tensor_parallel_size=tensor_parallel_size,
			max_model_len=max_model_len,
		)

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
		if not texts:
			return []
		normalized_allowed = {self._normalize(l): l for l in labels}
		label_str = ", ".join(labels)
		sampling = SamplingParams(
			max_tokens=max_tokens,
			temperature=temperature,
			stop=["\n"],
		)
		prompts = []
		for t in texts:
			parts = []
			if system_hint:
				parts.append(system_hint.strip())
			if instruction:
				parts.append(instruction.strip())
			parts.append(f"Classify the following text into one of [{label_str}].\nText: {t}\nLabel:")
			prompts.append("\n\n".join(parts))
		outputs = self.engine.generate(prompts, sampling)
		results: List[Dict[str, float]] = []
		for out in outputs:
			generated_text = out.outputs[0].text if out.outputs else ""
			norm = self._normalize(generated_text)
			matched = None
			for key, canonical in normalized_allowed.items():
				if norm.startswith(key):
					matched = canonical
					break
			if matched is None:
				matched = labels[0]
			results.append({"label": matched, "score": 1.0})
		return results


