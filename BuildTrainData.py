import PraetorianServerConnect as ps
import hashlib
from pathlib import Path

def write_training_datafile(server):
    fname = hashlib.md5(server.binary).hexdigest()
    outdir = 'B:/ML_Data/Praetorian/' + server.ans + '/'
    if not Path(outdir).is_dir():
        Path(outdir).mkdir(parents=True, exist_ok=True)
    if Path(outdir + fname).is_file():
        return -1
    ff = open(outdir + fname, 'x')
    ff.write(str(server.binary))
    ff.close()
    return 1

def main():
    numMinSamples = 3000
    trainingDataCountMap = {'avr':0,
                            'alphaev56':0,
                            'arm':0,
                            'm68k':0,
                            'mips':0,
                            'mipsel':0,
                            'powerpc':0,
                            's390':0,
                            'sh4':0,
                            'sparc':0,
                            'x86_64':0,
                            'xtensa':0}
    sv = ps.Server()
    while min(list(trainingDataCountMap.values())) < numMinSamples:
        sv.get()
        sv.post(sv.targets[0])
        ch = write_training_datafile(sv)
        if ch > 0:
            trainingDataCountMap[sv.ans] += 1
            print(trainingDataCountMap)

if __name__ == "__main__":
    main()
