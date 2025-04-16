import numpy as np

from itertools import product
from dataclasses import dataclass, field
from tqdm import tqdm
from scipy import stats

CONV_ALPHABET = "ACGTN"
SEQ_MIN = np.array(["A"], dtype="S1").view(np.uint8)[0]
SEQ_TO_INT_ARR = np.full(26, -1, dtype=int)
SEQ_TO_INT_ARR[0] = 0
SEQ_TO_INT_ARR[2] = 1
SEQ_TO_INT_ARR[6] = 2
SEQ_TO_INT_ARR[19] = 3


def seq_to_int(seq):
    """Convert string sequence to integer encoded array

    Args:
        seq (str): Nucleotide sequence

    Returns:
        np.array containing integer encoded sequence
    """
    return SEQ_TO_INT_ARR[
        np.array(list(seq), dtype="c").view(np.uint8) - SEQ_MIN
    ]


def int_to_seq(np_seq, alphabet=CONV_ALPHABET):
    """Convert integer encoded array to string sequence

    Args:
        np_seq (np.array): integer encoded sequence

    Returns:
        String nucleotide sequence
    """
    if np_seq.shape[0] == 0:
        return ""
    if np_seq.max() >= len(alphabet):
        tqdm.write(f"Invalid value in int sequence ({np_seq.max()})")
    return "".join(alphabet[b] for b in np_seq)


def index_from_kmer(kmer, alphabet="ACGT"):
    """Encode string k-mer as integer via len(alphabet)-bit encoding.

    Args:
        kmer (str): kmer string
        alphabet (str): bases used. Default: ACGT

    Returns:
        int: bit encoded kmer

    Example:
        index_from_kmer('AAA', 'ACG')               returns 0
        index_from_kmer('CAAAAAAAA', 'ACGTVWXYX')   returns 65536
    """
    return sum(
        alphabet.find(base) * (len(alphabet) ** kmer_pos)
        for kmer_pos, base in enumerate(kmer[::-1])
    )

def index_from_int_kmer(int_kmer, kmer_len):
    idx = 0
    for kmer_pos in range(kmer_len):
        idx += int_kmer[kmer_len - kmer_pos - 1]*(4**kmer_pos)
    return idx


def extract_levels(int_seq, int_kmer_levels, kmer_len, center_idx):
    levels = np.zeros(int_seq.shape[0], dtype=np.float32)
    for pos in range(int_seq.shape[0] - kmer_len + 1):
        center_pos = pos + center_idx
        kmer_idx = index_from_int_kmer(int_seq[pos : pos + kmer_len], kmer_len)
        levels[center_pos] = int_kmer_levels[kmer_idx]

    return levels

@dataclass
class kmerModel:
    kmer_model_filename: str = None

    _levels_array: np.ndarray = None
    str_kmer_levels: dict = None
    kmer_len: int = None
    center_idx: int = -1
    is_loaded: bool = False

    def __repr__(self):
        if not self.is_loaded:
            return "No Remora signal refine/map settings loaded"
        r_str = (
            f"Loaded {self.kmer_len}-mer table with {self.center_idx + 1} "
            "central position."
        )
        return r_str

    @property
    def bases_before(self):
        """Number of bases in k-mer before the central base"""
        return self.center_idx

    @property
    def bases_after(self):
        """Number of bases in k-mer after the central base"""
        return self.kmer_len - self.center_idx - 1

    def write_kmer_table(self, fh):
        for kmer in product(*["ACGT"] * self.kmer_len):
            fh.write(
                f"{''.join(kmer)}\t"
                f"{self._levels_array[index_from_kmer(kmer)]}\n"
            )

    def load_kmer_table(self):
        self.str_kmer_levels = {}
        with open(self.kmer_model_filename) as kmer_fp:
            self.kmer_len = len(kmer_fp.readline().split()[0])
            kmer_fp.seek(0)
            for line in kmer_fp:
                kmer, level = line.split()
                kmer = kmer.upper()
                if kmer in self.str_kmer_levels:
                    tqdm.write(
                        f"K-mer found twice in levels file '{kmer}'."
                    )
                if self.kmer_len != len(kmer):
                    tqdm.write(
                        f"K-mer lengths not all equal '{len(kmer)} != "
                        f"{self.kmer_len}' for {kmer}."
                    )
                try:
                    self.str_kmer_levels[kmer] = float(level)
                    if np.isnan(self.str_kmer_levels[kmer]):
                        self.str_kmer_levels[kmer] = 0
                except ValueError:
                    tqdm.write(
                        f"Could not convert level to float '{level}'"
                    )
        if len(self.str_kmer_levels) != 4 ** self.kmer_len:
            tqdm.write(
                "K-mer table contains fewer entries "
                f"({len(self.str_kmer_levels)}) than expected "
                f"({4 ** self.kmer_len})"
            )

    def determine_dominant_pos(self):
        if self.str_kmer_levels is None:
            return
        sorted_kmers = sorted(
            (level, kmer) for kmer, level in self.str_kmer_levels.items()
        )
        kmer_idx_stats = []
        kmer_summ = ""
        for kmer_idx in range(self.kmer_len):
            kmer_idx_pos = []
            for base in "ACGT":
                kmer_idx_pos.append(
                    [
                        levels_idx
                        for levels_idx, (_, kmer) in enumerate(sorted_kmers)
                        if kmer[kmer_idx] == base
                    ]
                )
            # compute Kruskal-Wallis H-test statistics for non-random ordering
            # of groups, indicating the dominant position within the k-mer
            kmer_idx_stats.append(stats.kruskal(*kmer_idx_pos)[0])
            kmer_summ += f"\t{kmer_idx}\t{kmer_idx_stats[-1]:10.2f}\n"
        self.center_idx = np.argmax(kmer_idx_stats)
        tqdm.write(f"K-mer index stats:\n{kmer_summ}")
        tqdm.write(f"Choosen central position: {self.center_idx}")

    @property
    def levels_array(self):
        if self._levels_array is None:
            if self.str_kmer_levels is None:
                return
            self._levels_array = np.empty(4 ** self.kmer_len, dtype=np.float32)
            for kmer, level in self.str_kmer_levels.items():
                self._levels_array[index_from_kmer(kmer)] = level
        return self._levels_array

    def __post_init__(self):
        # determine if level model is loaded from any of 3 options
        if self._levels_array is not None and not np.array_equal(
                self._levels_array, np.array(None)
        ):
            self.is_loaded = True
            self.kmer_len = int(np.log(self._levels_array.size) / np.log(4))
            assert 4 ** self.kmer_len == self._levels_array.size
        elif self.kmer_model_filename is not None:
            self.load_kmer_table()
            self.is_loaded = True
            self.determine_dominant_pos()
        elif self.str_kmer_levels is not None:
            self.is_loaded = True
            self.determine_dominant_pos()


    def extract_levels(self, int_seq):
        return extract_levels(
            int_seq.astype(np.int32),
            self.levels_array,
            self.kmer_len,
            self.center_idx,
        )

    @classmethod
    def load_from_metadata(cls, metadata):
        return cls(
            _levels_array=metadata.get("refine_kmer_levels"),
            center_idx=metadata.get("refine_kmer_center_idx")
        )

    @classmethod
    def load_from_dict(
            cls,
            data,
            do_rough_rescale=True,
            scale_iters=-1,
            sd_params=None,
            do_fix_guage=False
    ):
        """Create refiner from str_kmer_levels dict with kmer keys and float
        current level values.
        """
        kmer_len = len(next(iter(data.keys())))
        return cls(
            str_kmer_levels=data,
            kmer_len=kmer_len
        )

    def __eq__(self, other):
        if not isinstance(other, kmerModel):
            return False
        if (
                not np.array_equal(self._levels_array, other._levels_array)
                or self.center_idx != other.center_idx
        ):
            return False
