import logging
from dataclasses import dataclass
from collections import defaultdict
from typing import Generator
from pysam import AlignedSegment, AlignmentFile
import pickle

@dataclass
class BamIndex:
    '''
    A mapping from (read id -> pointer to aligned segment).
    '''

    bam_path: str
    logger: logging.Logger = logging.getLogger('bam_index')

    def __post_init__(self) -> None:
        self.bam_file: AlignmentFile = None
        self.num_recs: int = 0
        self.aligned: bool = False
        self.build_index()

    def open_bam(self) -> None:
        self.bam_file = AlignmentFile(self.bam_path, 'rb', check_sq=False)

    def close_bam(self) -> None:
        self.bam_file.close()
        self.bam_file = None

    def build_index(self) -> None:
        try: # load from disk if possible 
            with open(f'./bam_cache.pkl', 'rb') as f: 
                self.logger.info('loading cached BAM file...')
                self.bam_idx = pickle.load(f)
                self.logger.info('loading cached BAM file done.')
            return 
        except: pass

        if self.bam_file is None: self.open_bam()

        #self.bam_idx = defaultdict(list)
        self.bam_idx = defaultdict(int)
        self.logger.info('cached index not found -- indexing BAM file by read ids...')

        while True:
            read_ptr = self.bam_file.tell()

            try: read = next(self.bam_file)
            except StopIteration:
                self.logger.info('finished reading bam file.')
                break

            read_id = read.query_name
            if read.is_supplementary or read.is_secondary or read_id in self.bam_idx: continue

            self.num_recs += 1
            #self.bam_idx[read_id].append(read_ptr)
            self.bam_idx[read_id] = read_ptr

        self.close_bam()
        self.bam_idx = dict(self.bam_idx)
        self.num_reads = len(self.bam_idx.keys())
        self.logger.info(f'number of reads: {self.num_reads}')

        with open(f'./bam_cache.pkl', 'xb') as f: # write to disk
            pickle.dump(self.bam_idx, f)
            self.logger.info(f'cached BAM index: bam_cache.pkl')

    #def get_alignment(self, read_id: str) -> Generator[AlignedSegment, None, None]:
    def get_alignment(self, read_id: str) -> AlignedSegment:
        if self.bam_file is None: self.open_bam()

        #try: read_ptrs = self.bam_idx[read_id]
        #except KeyError:
            #self.logger.warning(f'cannot find read {read_id} in bam index.')
            #return None

        try: pointer = self.bam_idx[read_id]
        except KeyError:
            #self.logger.warning(f'cannot find read {read_id} in bam index.')
            return None

        self.bam_file.seek(pointer)

        try: aligned_segment = next(self.bam_file)
        except OSError:
            self.logger.error(f'cannot extract read {read_id} from BAM index.')
            return None

        if str(aligned_segment.query_name) != read_id:
            raise ValueError(f'read id {read_id} doesnt match read retrieved from index {aligned_segment.query_name}.')

        return aligned_segment
        #yield aligned_segment

        #for read_ptr in read_ptrs:
            #self.bam_file.seek(read_ptr)

            #try: bam_read = next(self.bam_file)
            #except OSError:
                #self.logger.warning(f'cannot extract read {read_id} from BAM index.')
                #continue

            #assert str(bam_read.query_name) == read_id, \
                #self.logger.error(f'read id {read_id} doesnt match read retrieved from index {bam_read.query_name}.')

            #yield bam_read
