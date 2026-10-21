"""Global constants shared by every module.

Everything that defines the experimental protocol (sampling rate, window length,
unified label set, patient splits) lives here so results are reproducible.
"""
FS = 250                       # common sampling rate after resampling (Hz)
WIN_S = 10                     # window length in seconds
WIN = FS * WIN_S               # samples per window (2500)

CLASSES = ["NSR", "SBRAD", "STACH", "AF", "PVC"]
NSR, SBRAD, STACH, AF, PVC = range(5)
N_CLASSES = len(CLASSES)
OTHER = -1                     # windows outside the five classes (OOD analysis only)

# --- MIT-BIH -----------------------------------------------------------------
# Records with paced rhythm are excluded from classification (standard practice).
MITBIH_PACED = {"102", "104", "107", "217"}
# Inter-patient split of de Chazal et al. (2004): DS1 train, DS2 test.
MITBIH_DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
              "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
              "223", "230"]
MITBIH_DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
              "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
              "233", "234"]
# Validation records are held out of DS1 (so validation patients are never trained on).
MITBIH_VAL = ["106", "124", "203", "209"]
MITBIH_TRAIN = [r for r in MITBIH_DS1 if r not in MITBIH_VAL]

# Annotation symbols that are real heart beats (used for R-R intervals / PVC detection).
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")
PVC_SYMBOLS = {"V"}
PACED_SYMBOLS = {"/", "f"}
AF_RHYTHMS = {"(AFIB", "(AFL"}
NORMAL_RHYTHMS = {"(N", "(SBR"}   # '(SBR' = sinus bradycardia rhythm annotation

# Heart-rate bounds (bpm) that define the rate-based classes.
BRADY_BPM = 60
TACHY_BPM = 100

# --- PTB-XL ------------------------------------------------------------------
PTBXL_VERSION = "1.0.3"
PTBXL_TRAIN_FOLDS = list(range(1, 9))
PTBXL_VAL_FOLD = 9
PTBXL_TEST_FOLD = 10

# --- Signal-quality / noise ----------------------------------------------------
SNR_LEVELS = [24, 18, 12, 6, 0, -6]
NOISE_KINDS = ["bw", "em", "ma"]          # baseline wander, electrode motion, muscle artifact
NOISE_NAMES = {"bw": "Baseline wander", "em": "Electrode motion", "ma": "Muscle artifact"}
