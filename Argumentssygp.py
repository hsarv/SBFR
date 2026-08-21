from sys import argv
STAN_APPROACH = False
RUNS = 1
MAX_Evaluation =500
POPULATION_SIZE = 200
UPDATEALL = False
VALIDATION = False
KFOLD =3 #2 #3
OPERATORS = [("+", 2), ("-", 2), ("*", 2), ("/", 2)]  # Default
#
# OPERATORS = [
#     ("+", 2),   # addition
#     ("-", 2),   # subtraction
#     ("*", 2),   # multiplication
#     ("/", 2),   # division
#     ("log", 1), # natural log
#     ("exp", 1), # exponential
#     ("pow", 2), # power(base, exponent)
#
# ]
# OPERATORS = [
#     ("+", 2),   # addition
#     ("-", 2),   # subtraction
#     ("*", 2),   # multiplication
#     ("/", 2),   # division
#     ("log", 1), # natural log
#     ("log2", 1),# log base 2
#     ("exp", 1), # exponential
#     ("pow", 2), # power(base, exponent)
#     ("max", 2), # max of 2 (can be extended to n-ary)
#     ("min", 2), # min of 2 (can be extended to n-ary)
# ]
MAX_DEPTH = 6
MAX_NICHE_SIZE = 300
MAX_NICHE_COUNT = 25
MAX_GENERATION = 10
TRAIN_FRACTION = 0.70
ELITISM_SIZE = 1
SHUFFLE = True
dataset_name = "banana"
LIMIT_DEPTH = 12
VERBOSE = True
# Number of CPU Threads to be used
THREADS = 1
# Minimum number of dimensions
DIM_MIN = 1
# An unreachable number of dimensions
DIM_MAX = 1
# Random state
RANDOM_STATE = 42
FITNESS_TYPE = ["Accuracy","MSE", "WAF", "CrossEntropy", "2FOLD",]
DATASETS_DIR = "datasets/"
OUTPUT_DIR = "results"
OUTPUT = "Classification"

if "-dsdir" in argv:
	DATASETS_DIR = argv[argv.index("-dsdir")+1]

if "-odir" in argv:
	OUTPUT_DIR = argv[argv.index("-odir")+1]

if "-d" in argv:
	DATASETS = argv[argv.index("-d")+1].split(";")

if "-runs" in argv:
	RUNS = int(argv[argv.index("-runs")+1])

if "-op" in argv:
	OPERATORS = argv[argv.index("-op")+1].split(";")
	for i in range(len(OPERATORS)):
		OPERATORS[i] = OPERATORS[i].split(",")
		OPERATORS[i][1] = int(OPERATORS[i][1])

if "-md" in argv:
	MAX_DEPTH = int(argv[argv.index("-md")+1])

if "-ps" in argv:
	POPULATION_SIZE = int(argv[argv.index("-ps")+1])

if "-mg" in argv:
	MAX_GENERATION = int(argv[argv.index("-mg")+1])

if "-tf" in argv:
	TRAIN_FRACTION = float(argv[argv.index("-tf")+1])

if "-ts" in argv:
	TOURNAMENT_SIZE = int(argv[argv.index("-ts")+1])

if "-es" in argv:
	ELITISM_SIZE = int(argv[argv.index("-es")+1])

if "-dontshuffle" in argv:
	SHUFFLE = False

if "-s" in argv:
	VERBOSE = False

if "-t" in argv:
	THREADS = int(argv[argv.index("-t")+1])

if "-dmin" in argv:
	DIM_MIN = int(argv[argv.index("-dmin")+1])

if "-dmax" in argv:
	DIM_MAX = int(argv[argv.index("-dmax")+1])


if "-rs" in argv:
	RANDOM_STATE = int(argv[argv.index("-rs")+1])


