from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


def collect_snippets(base_dir, arch_dir):
    arch_path = base_dir + arch_dir + '/'
    snippet_list = list()
    filelist = Path(arch_path).glob('*')
    for ff in filelist:
        bytestr = ff.read_text()
        bytestr = bytestr[2:-2]
        snippet_list.append(bytestr)
    return snippet_list

def build_unique_wordlist(arch_dict, max_words):
    uniq_wordlist = list()
    all_snippets = arch_dict.values()
    for arch in arch_dict.keys():
        all_snippets = arch_dict[arch]
        for snippet in all_snippets:
            split_bytestr = snippet.split('\\x')
            if len(split_bytestr) > max_words:
                max_words = len(split_bytestr)
            for byte in split_bytestr:
                if not byte.strip() and not byte.strip() in uniq_wordlist:
                    uniq_wordlist.append(byte.strip())
                elif not byte.strip():
                    continue
                uniq_wordlist.append(byte.strip())
    wordlist = Counter(uniq_wordlist)
    trimmed_wordlist = list()
    for key, val in wordlist.items():
        if val > 0.0001 * len(uniq_wordlist):
            trimmed_wordlist.append(key)


    uniq_wordlist = sorted(list(set(trimmed_wordlist)))
    return uniq_wordlist, max_words

def get_byte_occurance_rates(arch_dict, unique_bytelist):
    arch_list = sorted(arch_dict.keys())

    binned_arch_dict = dict()
    for arch in arch_list:
        binned_arch_dict[arch] = np.zeros((len(unique_bytelist)))
        total_byte_count = 0
        for snips in arch_dict[arch]:
            split_bytestr = snips.split('\\x')
            for byte in split_bytestr:
                try:
                    unique_bytelist.index(byte.strip())
                except ValueError:
                    continue
                total_byte_count += 1
                binned_arch_dict[arch][unique_bytelist.index(byte.strip())] += 1
        binned_arch_dict[arch] = binned_arch_dict[arch]/total_byte_count
    return binned_arch_dict

def main():
    training_data_basedir = 'B:/ML_Data/Praetorian/'
    outdir = 'B:/ML_Data/Praetorian_stats/'
    fname = 'byte_stats_trimmed1.csv'
    max_words=42
    
    arch_dict = dict()
    # scan folder, get archs
    arch_set = sorted(Path(training_data_basedir).glob('*'))

    # for each arch, collect words & process
    for arch in arch_set:
        arch_str = str(arch).split('\\')
        arch_str = arch_str[-1]
        arch_dict[str(arch_str)] = collect_snippets(training_data_basedir, str(arch_str))
    wordlist, max_words = build_unique_wordlist(arch_dict, max_words)
    print('Byte list length: %i' %(len(wordlist)))
    arch_list = sorted(list(arch_dict.keys()))
    

    binned_arch_dict = get_byte_occurance_rates(arch_dict, wordlist)
    arch_df = pd.DataFrame.from_dict(binned_arch_dict, orient='columns')
    arch_df = arch_df.set_axis(wordlist, axis = 0)
    
    if not Path(outdir).is_dir():
        Path(outdir).mkdir(parents=True, exist_ok=True)
    
    arch_df.to_csv(Path(outdir + fname))

if __name__ == "__main__":
    main()
