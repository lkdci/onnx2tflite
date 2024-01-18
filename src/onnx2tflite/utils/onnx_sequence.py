from typing import List, Dict
from abc import ABC, abstractmethod

from onnx import NodeProto


class IONNXSequence(ABC):
    @classmethod
    @abstractmethod
    def op_types(cls) -> List[str]:
        raise NotImplementedError

    @classmethod
    def op_counts(cls) -> Dict[str, int]:
        return cls.get_op_counts(cls.op_types())

    @classmethod
    def num_ops(cls):
        return len(cls.op_types())

    @classmethod
    def is_sequence(cls, nodes: List[NodeProto], index: int, strict_order: bool = True) -> bool:
        # First quick check, if model doesn't contains enough nodes for the sequence.
        if cls.num_ops() > len(nodes) - index:
            return False
        # Second check looks for the same operators count in the following layers
        _op_types = [nodes[i].op_type for i in range(index, index + cls.num_ops())]
        _op_counts = cls.get_op_counts(_op_types)
        if _op_counts != cls.op_counts():
            return False
        # Third check strict that the order of ops is the same.
        if strict_order:
            for _op_type, expected_op_type in zip(_op_types, cls.op_types()):
                if _op_type != expected_op_type:
                    return False
        return True

    @staticmethod
    def get_op_counts(op_types: List[str]) -> Dict[str, int]:
        _op_counts = {}
        for _type in op_types:
            if _type not in _op_counts:
                _op_counts[_type] = 0
            _op_counts[_type] += 1
        return _op_counts
