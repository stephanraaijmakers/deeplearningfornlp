import sys
import re
import glob

def main(fn_regex):
    for fn in glob.glob(fn_regex):
        fn_out=fn.replace(".py","_no_comment_tags.py")
        print(fn, " => ",fn_out)
        fp=open(fn,"r")
        fp_out=open(fn_out,"w")
        for line in fp:
            new_line=re.sub("\<\d+\>","",line.rstrip())
            print(re.sub("\[CA\]","",new_line),file=fp_out)
        fp.close()
        fp_out.close()

if __name__=="__main__":
    main(sys.argv[1]) # file regexex, use double quotes, like "*.py"


    
    
     
