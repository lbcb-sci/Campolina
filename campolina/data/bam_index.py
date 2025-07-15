import logging
from dataclasses import dataclass
from collections import defaultdict
from typing import Generator
from pysam import AlignedSegment, AlignmentFile

@dataclass
class BamIndex:
    bampath: str
    logger = logging.getLogger('bam_index')

    def __post_init__(self):
        self.bam_f = None
        self.num_recs = 0
        self.aligned = False
        self.build_index()

    def open_bam(self):
        self.bam_f = AlignmentFile(self.bampath, 'rb', check_sq=False)

    def close_bam(self):
        self.bam_f.close()
        self.bam_f = None

    def build_index(self):
        if self.bam_f is None: self.open_bam()

        self.bam_idx = defaultdict(list)
        self.logger.info('indexing BAM file by read ids...')

        while True:
            read_ptr = self.bam_f.tell()

            try: read = next(self.bam_f)
            except StopIteration:
                self.logger.info('finished reading bam file.')
                break

            read_id = read.query_name
            if read.is_supplementary or read.is_secondary or read_id in self.bam_idx: continue

            self.num_recs += 1
            self.bam_idx[read_id].append(read_ptr)

        self.close_bam()
        self.bam_idx = dict(self.bam_idx)
        self.num_reads = len(self.bam_idx)

    def get_alignment(self, read_id: str) -> Generator[AlignedSegment, None, None]:
        if self.bam_f is None: self.open_bam()

        try: read_ptrs = self.bam_idx[read_id]
        except KeyError:
            self.logger.warning(f'cannot find read {read_id} in bam index.')
            return None

        for read_ptr in read_ptrs:
            self.bam_f.seek(read_ptr)

            try: bam_read = next(self.bam_f)
            except OSError:
                self.logger.warning(f'cannot extract read {read_id} from BAM index.')
                continue

            assert str(bam_read.query_name) == read_id, \
                self.logger.error(f'read id {read_id} doesnt match read retrieved from index {bam_read.query_name}.')

            yield bam_read
