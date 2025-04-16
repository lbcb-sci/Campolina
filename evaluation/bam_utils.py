import pysam

from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict
from pysam import AlignedSegment

@dataclass
class BamIndex:
    bampath: str

    def __post_init__(self):
        self.bam_f = None
        self.num_recs = 0
        self.aligned = False
        self.build_index()

    def open_bam(self):
        self.bam_f = pysam.AlignmentFile(self.bampath, 'r', check_sq=False)

    def close_bam(self):
        self.bam_f.close()
        self.bam_f = None

    def build_index(self):
        if self.bam_f is None:
            self.open_bam()
        self.bam_idx = defaultdict(list)
        tqdm.write('Indexing BAM file by read ids')

        while True:
            read_ptr = self.bam_f.tell()
            try:
                read = next(self.bam_f)
            except StopIteration:
                tqdm.write('Finished reading bam file')
                break
            read_id = read.query_name
            if read.is_supplementary or read.is_secondary or read_id in self.bam_idx:
                continue
            self.num_recs += 1
            self.bam_idx[read_id].append(read_ptr)
        self.close_bam()
        self.bam_idx = dict(self.bam_idx)
        self.num_reads = len(self.bam_idx)

    def get_alignment(self, read_id: str) -> AlignedSegment:
        if self.bam_f is None:
            self.open_bam()
        try:
            read_ptrs = self.bam_idx[read_id]
        except KeyError:
            tqdm.write(f'Cannot find read {read_id} in bam index')
            yield None
        for read_ptr in read_ptrs:
            self.bam_f.seek(read_ptr)
            try:
                bam_read = next(self.bam_f)
            except OSError:
                tqdm.write(f'Cannot extract read {read_id} from bam index')
                continue
            assert str(bam_read.query_name) == read_id, (tqdm.write(f'Given read id {read_id} does not match read retrieved '
                                                               f'from bam index {bam_read.query_name}'))
            yield bam_read