import rdkit
import pandas as pd
from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import time, math
import psutil
# Disables the C++ errors in RDKIT when forming mols
RDLogger.DisableLog('rdApp.*')

# 7.5 Gig - start
# 16.4 Gig - end
# 9GIGs

RADIUS = 3
FPSIZE = 512
SAMPLE_SMILE = "c1cc(ccc1[C@H]([C@@H](CO)NC(=O)C(Cl)Cl)O)[N+](=O)[O-]"
RETURN_NUM_RESULTS = 200
SAMPLE_NAME = "result_files/CPU_fp_search_small_debug"
CSV_REF_DB = "files/chembl_csv.csv"
TOTAL = 2372673
DIVISION_SIZE = 100000
divided =  math.ceil(TOTAL / DIVISION_SIZE)
ROW_NUMBER = DIVISION_SIZE - 1
smile_from_db = None
ref_fps = None


def prepare_smiles() -> list:
    # Takes a CSV and looks for a column named smiles
    start = time.time()
    #db_smile_list = pd.read_csv(CSV_REF_DB, delimiter="\t", names=["smile"])
    db_smile_list = pd.read_csv(CSV_REF_DB, usecols=["smile"], low_memory=True)
    smile_list = []
    cleaned_mols = []
    for x in db_smile_list['smile']:
        mol = Chem.MolFromSmiles(x)
        if mol:
            cleaned_mols.append(mol)
            smile_list.append(x)
    # Getting % usage of virtual_memory ( 3rd field)
    print('RAM memory % used:', psutil.virtual_memory()[2])
    # Getting usage of virtual_memory in GB ( 4th field)
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    del db_smile_list
    # cleaned_mols = [x for x in mols if x is not None]
    # smile_list = [Chem.MolToSmiles(x) for x in cleaned_mols]
    # smile_list_df = pd.DataFrame(smile_list)
    end = time.time()
    print(f"{end - start} seconds to read in {len(smile_list)} files")
    return cleaned_mols, smile_list

def prepare_smiles_low_memory() -> list:
    #Go look at the last line of the smiles -> 2,372,673 is the last index
    skip_rows = lambda x : x * ROW_NUMBER if x == 0 else x * ROW_NUMBER
    for i in range(0, divided): 
        this = pd.read_csv(CSV_REF_DB, skiprows= skip_rows(i), nrows=ROW_NUMBER, names = ['smile'])
        this.to_parquet(f"pq/pq_{i}.parquet", index=None)

def read_parque_low_memory():
    smile_list = []
    ref_fps = []
    for i in range(0, divided):
        this = pd.read_parquet(f"pq/pq_{i}.parquet", columns=['smile'])
        for x in this['smile']:
            #Sometimes thes fail to generate so need to test
            mol = Chem.MolFromSmiles(x)
            if mol:
                ref_fps.append(generate_fps_indiv_mol(mol))
                smile_list.append(x)

    return ref_fps, smile_list


def generate_fps_list(cleaned_mols: list) -> list("ExplicitBitVect"):
    # Generate FP for db smiles
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS, fpSize=FPSIZE)
    return [mfpgen.GetFingerprint(m) for m in cleaned_mols]

def generate_fps_indiv_mol(mol: str) -> list("ExplicitBitVect"):
    # Generate FP for submitted/sample smile
    mfpgen_example = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS, fpSize=FPSIZE)
    return mfpgen_example.GetFingerprint(mol)

def generate_fps_indiv(sample_smile: str) -> list("ExplicitBitVect"):
    # Generate FP for submitted/sample smile
    mol = Chem.MolFromSmiles(sample_smile)
    mfpgen_example = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS, fpSize=FPSIZE)
    return mfpgen_example.GetFingerprint(mol)


def gen_tanimoto(sample_fps, ref_fps) -> list("tanimoto_Results"):
    # perform tanimoto
    rdkit.DataStructs.cDataStructs.BulkTanimotoSimilarity(sample_fps, ref_fps)
    return rdkit.DataStructs.cDataStructs.BulkTanimotoSimilarity(sample_fps, ref_fps)


def results_to_df(tanimoto_results, smile_from_db) -> dict:
    # Convert results to numpy
    result_df = pd.DataFrame(tanimoto_results, columns=["similarity"])
    result_df["smiles"] = smile_from_db
    return result_df.sort_values(by="similarity", ascending=False).head(RETURN_NUM_RESULTS).round(2).to_dict('records')


def return_stats() -> dict:
    stats = {
        "radius": RADIUS,
        "fpsize": FPSIZE
    }
    return stats


def startup():
    cleaned_mols, smile_from_db = prepare_smiles()
    ref_fps = generate_fps_list(cleaned_mols)
    return smile_from_db, ref_fps


def main(smile_from_db, ref_fps, SAMPLE_SMILE=None) -> pd.DataFrame:
    sample_fps = generate_fps_indiv(SAMPLE_SMILE)
    tanimoto_results = gen_tanimoto(sample_fps, ref_fps)
    results = results_to_df(tanimoto_results, smile_from_db)
    return results

def startup_low_memory():
    prepare_smiles_low_memory()
    ref_fps, smile_from_db = read_parque_low_memory()
    return smile_from_db, ref_fps

def main_low_memory(smile_from_db, ref_fps, smile) -> pd.DataFrame:
    print(smile)
    sample_fps = generate_fps_indiv(smile)
    tanimoto_results = gen_tanimoto(sample_fps, ref_fps)
    results = results_to_df(tanimoto_results, smile_from_db)
    return results

def run_test():
    smile_from_db, ref_fps = startup_low_memory()
    result_dict = main_low_memory(smile_from_db, ref_fps)
    print(result_dict)

if __name__ == "__main__":
    run_test()
