import csv
import struct
import tempfile
import unittest
from pathlib import Path

from bin_to_csv import convert


class BinToCsvTest(unittest.TestCase):
    def test_raw16_converts_to_g(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / 'raw.bin'
            output = Path(td) / 'raw.csv'
            source.write_bytes(struct.pack('<hhh', 1000, -2000, 2048))
            count = convert(source, output, 'raw16', 16, 6660)
            with output.open(newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(count, 1)
            self.assertEqual(rows[0]['t'], '0.000000000000')
            self.assertAlmostEqual(float(rows[0]['ax']), 0.488)
            self.assertAlmostEqual(float(rows[0]['ay']), -0.976)
            self.assertAlmostEqual(float(rows[0]['az']), 0.999424)

    def test_float32_ms2_to_g(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / 'float.bin'
            output = Path(td) / 'float.csv'
            source.write_bytes(struct.pack('<fff', 9.80665, 0, -19.6133))
            convert(source, output, 'float32-ms2', None, 6660)
            with output.open(newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertAlmostEqual(float(rows[0]['ax']), 1.0, places=6)
            self.assertAlmostEqual(float(rows[0]['az']), -2.0, places=6)

    def test_fixed_record_with_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / 'framed.bin'
            output = Path(td) / 'framed.csv'
            source.write_bytes(b'HEAD' + struct.pack('<Ihhh', 100, 100, 0, 0))
            convert(source, output, 'raw16', 2, 6660, header_bytes=4, record_bytes=10, xyz_offset=4)
            with output.open(newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertAlmostEqual(float(rows[0]['ax']), 0.0061)


if __name__ == '__main__':
    unittest.main()
