def print_important_lines(fname):
    with open(fname, 'r') as fl:
        important = [
            'Total time:',
            'beams._evaluate_beam(',
            'nufft3d3(',
            'nufft2d3(',
            'ModelData.from_config(',
            'write_uvh5(',
        ]
        
        for line in fl:
            if any(imp in line for imp in important):
                print(line, end='')

def get_peak_mem(logfile):
    mem = 0
    with open(logfile, 'r') as fl:
        for line in fl:
            if "Memory usage" in line:
                _m = float(line.split(": ")[1].split(" G")[0].strip())
                if _m > mem:
                    mem = _m
    return mem

if __name__ == '__main__':
    from pathlib import Path

    d = Path("profiling")
    allfiles = sorted(d.glob("*-fftvis-*"))

    for fl in allfiles:
        skymod, gpu, nt, layout, code, version, hsim = fl.name.split("-")

        logdir = Path(f"logs/vis/{skymod}/nt17280-{nt[2:]}chunks-{layout}")
        if not logdir.exists():
            print(f"SKIPPING  {logdir} since it doesn't exist")
        else:
            logfile = sorted(logdir.glob("fch0001-ch000_*.out"))[-1]
            peakmem = get_peak_mem(logfile)
        
        label = f"SKY={skymod}, NTIMES={nt[2:]}, LAYOUT={layout}"
        
        print(label)
        print("-"*len(label))
        print("Peak Memory: ", peakmem, "GB")
        print_important_lines(fl)
        print()
        print()
