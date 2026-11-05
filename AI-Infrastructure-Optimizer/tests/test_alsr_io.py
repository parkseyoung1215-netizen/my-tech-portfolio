import csv,tempfile,unittest
from pathlib import Path
from src.alsr_io import write_union

class TestALSRExport(unittest.TestCase):
    def test_mixed_training_and_calibration_trace_columns_survive(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'mixed.csv';write_union(p,[{'phase':'train','layer':3},{'phase':'calibration','threshold':.15,'skip_layers':[4,6]}])
            with p.open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),2);self.assertEqual(rows[0]['layer'],'3');self.assertEqual(rows[1]['threshold'],'0.15');self.assertEqual(rows[1]['skip_layers'],'[4, 6]')

    def test_existing_file_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'old.csv';p.write_text('original')
            with self.assertRaises(FileExistsError):write_union(p,[{'a':1}])
            self.assertEqual(p.read_text(),'original')

if __name__=='__main__':unittest.main()
