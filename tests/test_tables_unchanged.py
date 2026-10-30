"""Ensure existing RULA A/B/C reference matrices remain byte-for-byte equivalent in content."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

MAIN = Path(__file__).resolve().parents[1] / '222.py'
EXPECTED={
 'RULA_TABLE_A_ROWS': 'd32cbaf2d6df6325b40c5c4c8b881377811e6aeef211b8718050a31a58585a85',
 'RULA_TABLE_B_ROWS': 'eac402b9eed5a8bf367d9efadd8f47a2e06d2774e215896945128a319930699a',
 'RULA_TABLE_C_ROWS': '41fe19d4c21aa3aaff76d6a10e427b6274da2ada09e04762b08e88717ad8d68b',
}
class TestTablesUnchanged(unittest.TestCase):
    def test_reference_matrices_preserved(self):
        tree=ast.parse(MAIN.read_text(encoding='utf8'))
        found={}
        for node in tree.body:
            if isinstance(node,ast.Assign):
                for t in node.targets:
                    if isinstance(t,ast.Name) and t.id in EXPECTED:
                        a=ast.literal_eval(node.value)
                        digest=hashlib.sha256(json.dumps(a,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                        found[t.id]=digest
        self.assertEqual(EXPECTED,found)

if __name__=='__main__': unittest.main()
