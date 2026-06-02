"""Filter jsonl sentences that exactly match pd_mp_test.txt source."""
import argparse, json

def read_test_src(path):
    s=set(); buf=[]
    for line in open(path,encoding='utf-8'):
        line=line.rstrip()
        if not line:
            if buf: s.add(''.join(c for c,_ in buf)); buf=[]
        else:
            x=line.split(); buf.append((x[0],x[-1]))
    if buf: s.add(''.join(c for c,_ in buf))
    return s

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--test-bmes",default="data/pd_mp_test.txt")
    args=ap.parse_args()
    test=read_test_src(args.test_bmes)
    n_in=n_out=n_filt=0
    with open(args.output,'w',encoding='utf-8') as fout:
        for line in open(args.input,encoding='utf-8'):
            n_in+=1
            try: obj=json.loads(line)
            except: continue
            if obj.get('source') in test:
                n_filt+=1; continue
            fout.write(line)
            n_out+=1
    print(f"in={n_in} out={n_out} filtered={n_filt} ({100*n_filt/n_in:.2f}%)")

if __name__=="__main__":
    main()
