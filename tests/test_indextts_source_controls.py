"""Regression tests for the externally installed IndexTTS controls.

These tests intentionally inspect the source with ``ast`` rather than importing
ComfyUI or IndexTTS.  Importing either stack would require a live ComfyUI
runtime and heavyweight model dependencies, while the regressions covered here
are source-plumbing mistakes.
"""
from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path


# This is an opt-in integration check against a separately installed ComfyUI
# custom node. Unit/CI runs have no ComfyUI checkout, GPU, or model assets.
_NODE_DIR_ENV = "TIMBREFORGE_INDEXTTS_NODE_DIR"


def _source_paths() -> tuple[Path, Path]:
    node_dir = os.environ.get(_NODE_DIR_ENV, "").strip()
    if not node_dir:
        raise unittest.SkipTest(
            f"set {_NODE_DIR_ENV} to the installed IndexTTS custom-node directory "
            "to enable these source integration checks"
        )
    root = Path(node_dir).expanduser()
    return root / "index-tts" / "indextts" / "infer_v2_5.py", root / "nodes.py"


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class(tree: ast.AST, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name!r} not found")


def _function(owner: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(owner):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


def _attr_path(node: ast.AST) -> tuple[str, ...] | None:
    """Return a simple dotted path, including subscripts as a wildcard."""
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _attr_path(node.value)
        return parent + (node.attr,) if parent else None
    if isinstance(node, ast.Subscript):
        parent = _attr_path(node.value)
        return parent + ("<subscript>",) if parent else None
    return None


def _call_named(node: ast.Call, *path: str) -> bool:
    return _attr_path(node.func) == path


def _literal(node: ast.AST):
    if isinstance(node, ast.Constant):
        return node.value
    raise AssertionError(f"expected a literal AST node, got {ast.dump(node)}")


def _dict_value(node: ast.Dict, key: str) -> ast.AST:
    for key_node, value_node in zip(node.keys, node.values):
        if key_node is not None and _literal(key_node) == key:
            return value_node
    raise AssertionError(f"dictionary key {key!r} not found")


def _dict_literal(node: ast.Dict) -> dict:
    return {_literal(k): v for k, v in zip(node.keys, node.values) if k is not None}


class IndexTTS25SourceControlsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        infer_path, nodes_path = _source_paths()
        if not infer_path.is_file():
            raise AssertionError(f"missing IndexTTS source: {infer_path}")
        if not nodes_path.is_file():
            raise AssertionError(f"missing Comfy node source: {nodes_path}")
        cls.infer_tree = _tree(infer_path)
        cls.nodes_tree = _tree(nodes_path)

    def test_infer_generator_uses_runtime_sampling_and_diffusion_values(self):
        """The generator must not silently replace caller controls with literals."""
        cls = _class(self.infer_tree, "IndexTTS2")
        infer_generator = _function(cls, "infer_generator")

        gpt_calls = [
            node
            for node in ast.walk(infer_generator)
            if isinstance(node, ast.Call)
            and _call_named(node, "self", "gpt", "inference_speech")
        ]
        self.assertEqual(len(gpt_calls), 1)
        do_sample = next(
            (kw.value for kw in gpt_calls[0].keywords if kw.arg == "do_sample"), None
        )
        self.assertIsInstance(do_sample, ast.Name)
        self.assertEqual(do_sample.id, "do_sample")

        cfm_calls = [
            node
            for node in ast.walk(infer_generator)
            if isinstance(node, ast.Call)
            and _call_named(node, "self", "s2mel", "models", "<subscript>", "inference")
        ]
        self.assertEqual(len(cfm_calls), 1)
        cfm_call = cfm_calls[0]
        self.assertTrue(
            any(isinstance(arg, ast.Name) and arg.id == "diffusion_steps" for arg in cfm_call.args),
            "CFM must receive the runtime diffusion_steps variable",
        )
        cfg_kw = next(
            (kw.value for kw in cfm_call.keywords if kw.arg == "inference_cfg_rate"), None
        )
        self.assertIsInstance(cfg_kw, ast.Name)
        self.assertEqual(cfg_kw.id, "inference_cfg_rate")

        # Guard against the previous regression: assignments in the synthesis
        # block that overwrite the parsed controls with hard-coded defaults.
        for node in ast.walk(infer_generator):
            targets = []
            if isinstance(node, ast.Assign):
                targets.extend(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets.append(node.target)
            for target in targets:
                if not isinstance(target, ast.Name) or target.id not in {
                    "diffusion_steps",
                    "inference_cfg_rate",
                }:
                    continue
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    value = node.value
                    if target.id == "diffusion_steps" and isinstance(value, ast.Constant) and value.value == 25:
                        self.fail("infer_generator overwrites diffusion_steps with 25")
                    if target.id == "inference_cfg_rate" and isinstance(value, ast.Constant) and value.value == 0.7:
                        self.fail("infer_generator overwrites inference_cfg_rate with 0.7")

        pop_names = set()
        for node in ast.walk(infer_generator):
            if not isinstance(node, ast.Call) or not _call_named(node, "generation_kwargs", "pop"):
                continue
            if node.args:
                try:
                    pop_names.add(_literal(node.args[0]))
                except AssertionError:
                    pass
        self.assertIn("diffusion_steps", pop_names)
        self.assertIn("inference_cfg_rate", pop_names)

    def test_node_schema_and_signature_keep_controls_optional(self):
        synth = _class(self.nodes_tree, "IndexTTS25Synthesize")
        input_types = _function(synth, "INPUT_TYPES")
        returns = [node.value for node in ast.walk(input_types) if isinstance(node, ast.Return)]
        self.assertTrue(returns, "INPUT_TYPES must return a schema dictionary")
        schema = next((value for value in returns if isinstance(value, ast.Dict)), None)
        self.assertIsNotNone(schema)
        optional = _dict_value(schema, "optional")
        self.assertIsInstance(optional, ast.Dict)
        optional_items = _dict_literal(optional)

        for name, kind, default, minimum, maximum in (
            ("diffusion_steps", "INT", 25, 1, 100),
            ("inference_cfg_rate", "FLOAT", 0.7, 0.0, 2.0),
        ):
            self.assertIn(name, optional_items)
            spec = optional_items[name]
            self.assertIsInstance(spec, ast.Tuple)
            self.assertEqual(_literal(spec.elts[0]), kind)
            self.assertIsInstance(spec.elts[1], ast.Dict)
            options = _dict_literal(spec.elts[1])
            self.assertEqual(_literal(options["default"]), default)
            self.assertEqual(_literal(options["min"]), minimum)
            self.assertEqual(_literal(options["max"]), maximum)

        synthesize = _function(synth, "synthesize")
        arg_names = [arg.arg for arg in synthesize.args.args]
        self.assertIn("diffusion_steps", arg_names)
        self.assertIn("inference_cfg_rate", arg_names)
        defaults = synthesize.args.defaults
        default_by_name = dict(zip(arg_names[-len(defaults):], defaults))
        self.assertEqual(_literal(default_by_name["diffusion_steps"]), 25)
        self.assertEqual(_literal(default_by_name["inference_cfg_rate"]), 0.7)

    def test_node_forwards_controls_only_inside_25_branch(self):
        synth = _class(self.nodes_tree, "IndexTTS25Synthesize")
        synthesize = _function(synth, "synthesize")

        matching_branches = []
        for node in ast.walk(synthesize):
            if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
                continue
            if not isinstance(node.test.left, ast.Name) or node.test.left.id != "model_version":
                continue
            if len(node.test.comparators) != 1 or _literal(node.test.comparators[0]) != "2.5":
                continue
            matching_branches.append(node)

        self.assertEqual(len(matching_branches), 1)
        branch = matching_branches[0]
        updates = [
            node
            for node in ast.walk(branch)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
        ]
        self.assertEqual(len(updates), 1)
        update = updates[0]
        keys = {keyword.arg for keyword in update.keywords if keyword.arg is not None}
        self.assertIn("diffusion_steps", keys)
        self.assertIn("inference_cfg_rate", keys)


if __name__ == "__main__":
    unittest.main()
