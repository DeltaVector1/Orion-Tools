from typing import Any, List, Tuple, Union


def _split_path(path: str) -> List[str]:
	return [p for p in path.split(".") if p]


def find_nodes_by_path(root: Any, path: str) -> List[Tuple[Any, Union[str, int], Any]]:
	"""
	Traverse the object using a dot-separated path and return a list of nodes
	(parent, key_or_index, value) that correspond to the final segment.

	- Supports dict and list traversal.
	- If the path segment points to a list, traversal continues into each element.
	- Returns empty list if path not found.
	"""
	segments = _split_path(path)
	if not segments:
		return []

	current: List[Tuple[Any, Union[str, int], Any]] = [(None, None, root)]
	for seg in segments:
		next_level: List[Tuple[Any, Union[str, int], Any]] = []
		for parent, key, node in current:
			if isinstance(node, dict):
				if seg in node:
					next_level.append((node, seg, node[seg]))
			elif isinstance(node, list):
				for idx, item in enumerate(node):
					next_level.append((node, idx, item))
		current = next_level
		if not current:
			return []
	return current


def extract_texts(root: Any, path: str) -> List[str]:
	"""
	Extract text values found at the given dot-separated path.
	- If the leaf is a str, returns [str].
	- If the leaf is a list of str, returns that list (flattened across all matched leaves).
	- Non-str values are ignored.
	"""
	nodes = find_nodes_by_path(root, path)
	texts: List[str] = []
	for parent, key, value in nodes:
		if isinstance(value, str):
			texts.append(value)
		elif isinstance(value, list):
			for item in value:
				if isinstance(item, str):
					texts.append(item)
	return texts


def set_texts(root: Any, path: str, new_texts: List[str]) -> bool:
	"""
	Replace text at the given path with new_texts.
	- If leaf is a str, uses the first element of new_texts.
	- If leaf is a list of str, replaces elements up to len(new_texts).
	Returns True if at least one replacement happened.
	"""
	nodes = find_nodes_by_path(root, path)
	if not nodes:
		return False

	replacements = 0
	text_iter = iter(new_texts)
	for parent, key, value in nodes:
		try:
			if isinstance(value, str):
				try:
					parent[key] = next(text_iter)
					replacements += 1
				except StopIteration:
					break
			elif isinstance(value, list):
				for i in range(len(value)):
					try:
						candidate = next(text_iter)
						if isinstance(value[i], str):
							value[i] = candidate
							replacements += 1
					except StopIteration:
						break
		except Exception:
			continue
	return replacements > 0


